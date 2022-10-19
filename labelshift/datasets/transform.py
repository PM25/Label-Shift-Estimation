from torchvision import transforms

mean, std = {}, {}
mean["cifar10"] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean["cifar100"] = [x / 255 for x in [129.3, 124.1, 112.4]]

std["cifar10"] = [x / 255 for x in [63.0, 62.1, 66.7]]
std["cifar100"] = [x / 255 for x in [68.2, 65.4, 70.4]]

def get_transform_by_name(dset_name="cifar10", train=False):
    dset_name = dset_name.lower()
    crop_size = 96 if dset_name == "stl10" else 32
    transform = get_transform(mean[dset_name], std[dset_name], crop_size, train)
    return transform

def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])