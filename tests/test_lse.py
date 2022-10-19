import numpy as np

from torchvision import transforms

from labelshift import LSE


def test_lse():
    ntrain = 10000  # number of training data
    ntest = 1000  # number of testing data
    w, h, c = 32, 32, 3  # width, height, channels
    num_classes = 10

    fake_train_data = np.random.randint(0, 256, ntrain * h * w * c, dtype=np.uint8).reshape((ntrain, h, w, c))
    fake_train_labels = np.random.choice(range(num_classes), ntrain, replace=True)
    fake_test_data = np.random.randint(0, 256, ntest * h * w * c, dtype=np.uint8).reshape((ntest, h, w, c))

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    lse_algs = ["BBSE", "RLLS", "MLLS"]
    calibrations = [None, "BCTS"]

    lse = LSE(
        num_classes=num_classes,
        num_train_iter=50,
        num_eval_iter=50,
        num_ensemble=2,
        overwrite=True,
        lse_algs=lse_algs,
        calibrations=calibrations,
        gpu=None,
        seed=0,
    )
    lse.train_base_models(fake_train_data, fake_train_labels, num_classes, train_transform, test_transform, seed=0)
    estimations = lse.estimate(fake_test_data, test_transform=test_transform, save_name="test", verbose=True)

    for lse_alg in lse_algs:
        for calibration in calibrations:
            method_name = f"{lse_alg}_{calibration}"
            assert method_name in estimations

            estim_target_dist = estimations[method_name]["estimation"]
            assert abs(sum(estim_target_dist) - 1) < 1e-7
