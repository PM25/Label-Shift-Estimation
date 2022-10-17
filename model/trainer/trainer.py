import contextlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import *

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .trainer_utils import DRWLoss, mixup_one_target, to_one_hot, EMA, Bn_Controller, ce_loss


class ModelTrainer:
    def __init__(self, net_builder, num_classes, num_eval_iter=1000, ema_m=0.999):
        """
        class DistributionEstimate contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            num_eval_iter: frequency of evaluation.
        """
        super(ModelTrainer, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.num_eval_iter = num_eval_iter

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.ema_m = ema_m
        self.ema_model = deepcopy(self.model)

        self.bn_controller = Bn_Controller()

        self.checkpoint = None

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        print(f"[!] data loader keys: {self.loader_dict.keys()}")

    def set_dataset(self, dset_dict):
        self.dset_dict = dset_dict
        print(f"[!] datasets keys: {self.dset_dict.keys()}")

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args):

        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()

        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # deferred reweighting loss
        lb_dist_np = np.zeros(args.num_classes)
        for _, y in self.dset_dict["train_lb"]:
            lb_dist_np[y] += 1
        drw_loss = DRWLoss(args, lb_dist_np, warm_iter=int(args.drw_warm * args.num_train_iter))

        assert args.num_train_iter >= self.num_eval_iter, "num_eval_iter is smaller than num_train_iter"

        # start training
        pbar = tqdm(total=self.num_eval_iter, desc="training", leave=False)
        for x_lb, y_lb in self.loader_dict["train_lb"]:

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            torch.cuda.synchronize()

            x_lb = x_lb.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                if args.use_mixup_drw:
                    y_lb = to_one_hot(y_lb, self.num_classes, args.gpu)
                    mixed_x, mixed_y, _ = mixup_one_target(x_lb, y_lb, args.gpu)
                    logits_x = self.model(mixed_x)
                    total_loss = drw_loss(self.it, logits_x, mixed_y, use_hard_labels=False)
                    total_loss = total_loss.mean()
                else:
                    logits_x = self.model(x_lb)
                    total_loss = ce_loss(logits_x, y_lb, reduction="mean")

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            torch.cuda.synchronize()

            self.it += 1
            pbar.update(1)

            if self.it % self.num_eval_iter == 0:
                tb_dict = {
                    "train/loss": total_loss.detach().cpu().item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                eval_dict = self.evaluate(self.loader_dict["val_lb"], args=args)
                tb_dict.update(eval_dict)

                if tb_dict["eval/top-1-acc"] > best_eval_acc:
                    best_eval_acc = tb_dict["eval/top-1-acc"]
                    best_it = self.it

                print(f"\n[{self.it}/{args.num_train_iter}] BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters, {tb_dict}")

                del tb_dict
                pbar.reset()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict["val_lb"]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for x, y in tqdm(eval_loader, desc="evaluating", leave=False):
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        self.ema.restore()
        self.model.train()
        return {"eval/loss": (total_loss / total_num).cpu().item(), "eval/top-1-acc": top1, "eval/top-5-acc": top5}

    @torch.no_grad()
    def get_logits(self, loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if loader is None:
            loader = self.loader_dict["val_lb"]

        y_true = []
        y_logits = []
        for x, y in tqdm(loader, desc="get logits", leave=False):
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            logits = self.model(x)
            y_true.extend(y.cpu().tolist())
            y_logits.extend(logits.cpu().numpy())
        self.ema.restore()
        self.model.train()
        return np.array(y_logits), np.array(y_true)

    def save_model(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = deepcopy(self.model)
        self.ema.restore()
        self.model.train()

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "it": self.it,
                "ema_model": ema_model.state_dict(),
            },
            save_path,
        )

        print(f"model saved: {save_path}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.it = checkpoint["it"]
        self.ema_model.load_state_dict(checkpoint["ema_model"])

        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        self.ema.load(self.ema_model)

        print(f"model loaded: {load_path}")


if __name__ == "__main__":
    pass
