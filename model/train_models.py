# import needed library
import json
import copy
import random
import shutil
import logging
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision import transforms

from model import ModelTrainer
from model.default_args import get_default_args
from get_method import get_lse_methods
from datasets import get_data_loader, BasicDataset, ResampleDataset, ResampleDataset
from utils import net_builder, get_optimizer, get_cosine_schedule_with_warmup, labels_to_dist


class TrainModels:
    def __init__(self, **kwargs):
        self.args = get_default_args(**kwargs)
        if Path(self.args.save_path).exists() and self.args.overwrite:
            shutil.rmtree(self.args.save_path)

    def set_labeled_data(self, lb_data, lb_targets, num_classes, train_transform=None, test_transform=None):
        self.lb_data = lb_data
        self.lb_targets = lb_targets
        self.num_classes = num_classes
        self.args.num_classes = num_classes
        self.train_transform = train_transform if train_transform is not None else transforms.ToTensor()
        self.test_transform = test_transform if test_transform is not None else transforms.ToTensor()

    def estimate(self, ulb_data, ulb_targets, seed=0):
        self.pre_check()

        random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()

        if seed is not None:
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            cudnn.deterministic = True

        cudnn.benchmark = True
        self.args.bn_momentum = 1.0 - 0.999

        # Construct Dataset & DataLoader
        lb_dset = ResampleDataset(self.lb_data, self.lb_targets, self.num_classes, self.train_transform, self.test_transform, onehot=False)
        ulb_dset = BasicDataset(ulb_data, ulb_targets, self.num_classes, self.test_transform, is_ulb=True, onehot=False)

        ulb_dist = labels_to_dist(ulb_dset.targets, self.num_classes)
        with np.printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format}):
            print(f"Target distribution: {ulb_dist}\n")

        logits_log = self.get_ensemble_logits(lb_dset, ulb_dset)
        estimations = self.apply_lse(logits_log, ulb_dist)

        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.random.set_rng_state(torch_random_state)

        logging.warning("Estimate is FINISHED")
        return estimations

    def pre_check(self):
        if not torch.cuda.is_available():
            raise Exception("ONLY GPU TRAINING IS SUPPORTED")

        if self.args.gpu is not None:
            warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")

    def get_ensemble_logits(self, lb_dset, ulb_dset):
        _net_builder = net_builder(
            self.args.net,
            self.args.net_from_name,
            {
                "first_stride": 2 if "stl" in self.args.dataset else 1,
                "depth": self.args.depth,
                "widen_factor": self.args.widen_factor,
                "leaky_slope": self.args.leaky_slope,
                "bn_momentum": self.args.bn_momentum,
                "dropRate": self.args.dropout,
                "use_embed": False,
            },
        )

        logits_log = {"val_logits": [], "val_targets": [], "ulb_logits": []}

        for idx, train_lb_dset, val_lb_dset in lb_dset.resample(self.args.num_val_per_class, self.args.num_ensemble, seed=self.args.seed):
            print(f"\nTraining [{idx}/{self.args.num_ensemble}] Model")
            model = ModelTrainer(_net_builder, self.num_classes, num_eval_iter=self.args.num_eval_iter, ema_m=self.args.ema_m)
            # SET Optimizer & LR Scheduler
            ## construct SGD and cosine lr scheduler
            optimizer = get_optimizer(model.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.args.num_train_iter, num_warmup_steps=self.args.num_train_iter * 0)
            ## set SGD and cosine lr
            model.set_optimizer(optimizer, scheduler)

            if self.args.gpu is not None:
                torch.cuda.set_device(self.args.gpu)
                model.model = model.model.cuda(self.args.gpu)
            else:
                model.model = torch.nn.DataParallel(model.model).cuda()

            model.ema_model = copy.deepcopy(model.model)

            loader_dict = {}
            dset_dict = {"train_lb": train_lb_dset, "val_lb": val_lb_dset, "ulb": ulb_dset}

            loader_dict["train_lb"] = get_data_loader(
                dset_dict["train_lb"],
                self.args.batch_size,
                data_sampler=self.args.train_sampler,
                num_iters=self.args.num_train_iter,
                num_workers=self.args.num_workers,
            )
            loader_dict["val_lb"] = get_data_loader(
                dset_dict["val_lb"], self.args.eval_batch_size, num_workers=self.args.num_workers, drop_last=False
            )
            loader_dict["ulb"] = get_data_loader(
                dset_dict["ulb"], self.args.eval_batch_size, num_workers=self.args.num_workers, drop_last=False
            )

            ## set DataLoader
            model.set_dataset(dset_dict)
            model.set_data_loader(loader_dict)

            save_model_path = Path(self.args.save_path) / "models" / f"model_{idx}.pt"
            if save_model_path.exists():
                model.load_model(save_model_path)
            else:
                # START TRAINING
                trainer = model.train
                trainer(self.args)
                if self.args.save_model:
                    model.save_model(save_model_path)

            if "ulb" in loader_dict:
                raw_val_outputs, val_targets = model.get_logits(loader_dict["val_lb"], args=self.args)
                raw_ulb_outputs, _ = model.get_logits(loader_dict["ulb"], args=self.args)
                logits_log["val_logits"].append(raw_val_outputs)
                logits_log["val_targets"].append(val_targets)
                logits_log["ulb_logits"].append(raw_ulb_outputs)

        return logits_log

    def apply_lse(self, logits_log, ulb_dist):
        # apply label shift estimation and save results
        if self.args.lse_algs is not None:
            assert (
                len(logits_log["ulb_logits"]) == len(logits_log["val_logits"]) == len(logits_log["val_targets"]) >= self.args.num_ensemble
            )
            ulb_logits = logits_log["ulb_logits"][: self.args.num_ensemble]
            val_logits = logits_log["val_logits"][: self.args.num_ensemble]
            val_targets = logits_log["val_targets"][: self.args.num_ensemble]

            print("Target distribution estimations:")
            estimations = {}
            names, estimators = get_lse_methods(self.args.lse_algs, self.args.calibrations, use_ensemble=True)
            for name, estimator in zip(names, estimators):
                estimator.fit(ulb_logits, val_logits, val_targets)
                est_target_dist = estimator.estim_target_dist
                mse = mean_squared_error(ulb_dist, est_target_dist)
                estimations[name] = {"estimation": est_target_dist.tolist(), "mse": mse}
                with np.printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format}):
                    print(f"{name}: {est_target_dist}, MSE: {mse:.5f}")

            save_est_path = Path(self.args.save_path) / "estimation.json"
            with open(save_est_path, "w") as f:
                json.dump(estimations, f, indent=4)

            logging.warning(f"Estimation Saved Successfully: {save_est_path}")

        return estimations
