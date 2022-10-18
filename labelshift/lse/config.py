from dataclasses import dataclass, field


@dataclass
class Config:
    save_path: str = "./saved_models/temp"
    save_model: bool = True
    overwrite: bool = True
    num_train_iter: int = 16000
    num_eval_iter: int = 1600
    num_val_per_class: int = 5
    batch_size: int = 64
    eval_batch_size: int = 1024
    ema_m: float = 0.999
    use_mixup_drw: bool = True
    drw_warm: float = 0.75
    optim: str = "SGD"
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 0.0005
    bn_momentum: float = 1.0 - 0.999
    amp: bool = False
    clip: float = 0
    net: str = "WideResNet"
    net_from_name: bool = False
    depth: int = 28
    widen_factor: int = 2
    leaky_slope: float = 0.1
    dropout: float = 0.0
    dataset: str = "custom"
    train_sampler: str = "RandomSampler"
    num_classes: int = 0
    num_workers: int = 1
    seed: int = 0
    gpu: int = 0
    lse_algs: list = field(default_factory=lambda: ["MLLS"])
    calibrations: list = field(default_factory=lambda: ["BCTS"])
    num_ensemble: int = 10


if __name__ == "__main__":
    config = Config(lse_algs=["BBSE", "RLLS", "MLLS"])
    print(config.num_ensemble)
    print(config.lse_algs)
    print(config)
