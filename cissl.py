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

from model import ModelTrainer
from datasets import Imbalanced_Dataset, get_data_loader
from utils import net_builder, over_write_args_from_file, get_optimizer, get_cosine_schedule_with_warmup, labels_to_dist
from get_method import get_lse_methods


def main(args):
    global best_acc1

    if not torch.cuda.is_available():
        raise Exception("ONLY GPU TRAINING IS SUPPORTED")

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU. This will completely disable data parallelism.")

    if Path(args.save_path).exists() and args.overwrite:
        shutil.rmtree(args.save_path)

    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True
    args.bn_momentum = 1.0 - 0.999

    # Construct Dataset & DataLoader
    dset = Imbalanced_Dataset(name=args.dataset, num_classes=args.num_classes, data_dir=args.data_dir)
    lb_dset, ulb_dset = dset.get_lb_ulb_dset(
        args.max_labeled_per_class,
        args.max_unlabeled_per_class,
        args.lb_imb_ratio,
        args.ulb_imb_ratio,
        args.imb_type,
        seed=args.seed,
    )
    ulb_dist = labels_to_dist(ulb_dset.targets, args.num_classes)
    with np.printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format}):
        print(f"Target distribution: {ulb_dist}\n")

    n_logits = 0
    save_logits_path = Path(args.save_path) / f"u{args.ulb_imb_ratio}" / f"logits.pt"

    if save_logits_path.exists():
        logits_log = torch.load(save_logits_path)
        n_logits = len(logits_log["ulb_logits"])

    if n_logits < args.num_ensemble:
        _net_builder = net_builder(
            args.net,
            args.net_from_name,
            {
                "first_stride": 2 if "stl" in args.dataset else 1,
                "depth": args.depth,
                "widen_factor": args.widen_factor,
                "leaky_slope": args.leaky_slope,
                "bn_momentum": args.bn_momentum,
                "dropRate": args.dropout,
                "use_embed": False,
            },
        )
        logits_log = get_ensemble_logits(args, _net_builder, lb_dset, ulb_dset)

        save_logits_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits_log, save_logits_path)
        logging.warning(f"Logits Saved Successfully: {save_logits_path}")

    # apply label shift estimation and save results
    if args.lse_algs is not None:
        assert len(logits_log["ulb_logits"]) == len(logits_log["val_logits"]) == len(logits_log["val_targets"]) >= args.num_ensemble
        ulb_logits = logits_log["ulb_logits"][: args.num_ensemble]
        val_logits = logits_log["val_logits"][: args.num_ensemble]
        val_targets = logits_log["val_targets"][: args.num_ensemble]

        print("Target distribution estimations:")
        estimations = {}
        names, estimators = get_lse_methods(args.lse_algs, args.calibrations, use_ensemble=True)
        for name, estimator in zip(names, estimators):
            estimator.fit(ulb_logits, val_logits, val_targets)
            est_target_dist = estimator.estim_target_dist
            mse = mean_squared_error(ulb_dist, est_target_dist)
            estimations[name] = {"estimation": est_target_dist.tolist(), "mse": mse}
            with np.printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format}):
                print(f"{name}: {est_target_dist}, MSE: {mse:.5f}")

        save_est_path = Path(args.save_path) / f"u{args.ulb_imb_ratio}" / "estimation.json"
        with open(save_est_path, "w") as f:
            json.dump(estimations, f, indent=4)

        logging.warning(f"Estimation Saved Successfully: {save_est_path}")

    logging.warning("Training is FINISHED")


def get_ensemble_logits(args, _net_builder, lb_dset, ulb_dset):
    logits_log = {"val_logits": [], "val_targets": [], "ulb_logits": []}

    for idx, train_lb_dset, val_lb_dset in lb_dset.resample(args.num_val_per_class, args.num_ensemble, seed=args.seed):
        print(f"\nTraining [{idx}/{args.num_ensemble}] Model")
        model = ModelTrainer(_net_builder, args.num_classes, num_eval_iter=args.num_eval_iter, ema_m=args.ema_m)
        # SET Optimizer & LR Scheduler
        ## construct SGD and cosine lr scheduler
        optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.num_train_iter, num_warmup_steps=args.num_train_iter * 0)
        ## set SGD and cosine lr
        model.set_optimizer(optimizer, scheduler)

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.model = model.model.cuda(args.gpu)
        else:
            model.model = torch.nn.DataParallel(model.model).cuda()

        model.ema_model = copy.deepcopy(model.model)

        loader_dict = {}
        dset_dict = {"train_lb": train_lb_dset, "val_lb": val_lb_dset, "ulb": ulb_dset}

        loader_dict["train_lb"] = get_data_loader(
            dset_dict["train_lb"],
            args.batch_size,
            data_sampler=args.train_sampler,
            num_iters=args.num_train_iter,
            num_workers=args.num_workers,
        )
        loader_dict["val_lb"] = get_data_loader(dset_dict["val_lb"], args.eval_batch_size, num_workers=args.num_workers, drop_last=False)
        loader_dict["ulb"] = get_data_loader(dset_dict["ulb"], args.eval_batch_size, num_workers=args.num_workers, drop_last=False)

        ## set DataLoader
        model.set_dataset(dset_dict)
        model.set_data_loader(loader_dict)

        save_model_path = Path(args.save_path) / "models" / f"model_{idx}.pt"
        if save_model_path.exists():
            model.load_model(save_model_path)
        else:
            # START TRAINING
            trainer = model.train
            trainer(args)
            if args.save_model:
                model.save_model(save_model_path)

        if "ulb" in loader_dict:
            raw_val_outputs, val_targets = model.get_logits(loader_dict["val_lb"], args=args)
            raw_ulb_outputs, _ = model.get_logits(loader_dict["ulb"], args=args)
            logits_log["val_logits"].append(raw_val_outputs)
            logits_log["val_targets"].append(val_targets)
            logits_log["ulb_logits"].append(raw_ulb_outputs)

    return logits_log


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")

    """
    Save Configuration
    """
    parser.add_argument("--save_path", type=str, default="./saved_models/label_shift")
    parser.add_argument("--save_model", type=str2bool, default=True, help="whether to save model")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="whether to overwrite exist models")

    """
    Training Configuration
    """
    parser.add_argument("--num_train_iter", type=int, default=2**15, help="total number of training iterations")
    parser.add_argument("--num_eval_iter", type=int, default=2 * 8, help="evaluation frequency")
    parser.add_argument("--max_labeled_per_class", type=int, default=1500, help="the maximum number of labeled data per class")
    parser.add_argument("--num_val_per_class", type=int, default=5, help="number of validations per class")
    parser.add_argument("--max_unlabeled_per_class", type=float, default=2, help="the maximum number of unlabeled data per class")
    parser.add_argument("--batch_size", type=int, default=64, help="total number of batch size of labeled data")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--use_mixup_drw", type=str2bool, default=True, help="whether to use mixup + deferred reweighting")
    parser.add_argument(
        "--drw_warm", type=float, default=0.75, help="deferred reweighting warm iterations out of total training iterations"
    )

    """
    Optimizer Configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--amp", type=str2bool, default=False, help="use mixed precision training or not")
    parser.add_argument("--clip", type=float, default=0)

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="WideResNet")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)

    """
    GPU Configurations
    """
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    """
    Imbalanced Configurations
    """
    parser.add_argument("--imb_type", type=str, default="long", help="long tailed or step imbalanced")
    parser.add_argument("--lb_imb_ratio", type=float, default=1, help="imbalance ratio of labeled data")
    parser.add_argument("--ulb_imb_ratio", type=float, default=1, help="imbalance ratio of unlabeled data")

    """
    Label Shift Estimation Configurations
    """
    parser.add_argument("--lse_algs", type=json.loads, default=None, help="list of label shift estimation methods to use")
    parser.add_argument("--calibrations", type=json.loads, default=None, help="list of calibration methods to use")
    parser.add_argument("--num_ensemble", type=int, default=10, help="number of subsets for training ensemble models")

    # config file
    parser.add_argument("--c", type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    main(args)
