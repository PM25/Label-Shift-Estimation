import json
import argparse


def get_default_args(**kwargs):
    parser = argparse.ArgumentParser(description="")

    """
    Save Configuration
    """
    parser.add_argument("--save_path", type=str, default="./saved_models/temp")
    parser.add_argument("--save_model", type=str2bool, default=True, help="whether to save model")
    parser.add_argument("--overwrite", type=str2bool, default=True, help="whether to overwrite exist models")

    """
    Training Configuration
    """
    parser.add_argument("--num_train_iter", type=int, default=16000, help="total number of training iterations")
    parser.add_argument("--num_eval_iter", type=int, default=1600, help="evaluation frequency")
    parser.add_argument("--max_labeled_per_class", type=int, default=1500, help="the maximum number of labeled data per class")
    parser.add_argument("--num_val_per_class", type=int, default=5, help="number of validations per class")
    parser.add_argument("--max_unlabeled_per_class", type=float, default=3000, help="the maximum number of unlabeled data per class")
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
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
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
    parser.add_argument("--dataset", type=str, default="custom")
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)

    """
    GPU Configurations
    """
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

    """
    Label Shift Estimation Configurations
    """
    parser.add_argument("--lse_algs", type=json.loads, default=["MLLS"], help="list of label shift estimation methods to use")
    parser.add_argument("--calibrations", type=json.loads, default=["BCTS"], help="list of calibration methods to use")
    parser.add_argument("--num_ensemble", type=int, default=10, help="number of subsets for training ensemble models")

    args = parser.parse_args()
    over_write_args_from_dict(args, kwargs)
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def over_write_args_from_dict(args, dic):
    if len(dic) == 0:
        return
    for k in dic:
        setattr(args, k, dic[k])


if __name__ == "__main__":
    args = get_default_args(lse_algs=["BBSE", "RLLS", "MLLS"])
    print(args.num_ensemble)
    print(args.lse_algs)
