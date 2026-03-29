import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF


class TrainTransform:
    def __init__(self, patch_sz, p_hflip=0.5, p_vflip=0.5):
        self.crop = transforms.RandomCrop(patch_sz)
        self.hflip = transforms.RandomHorizontalFlip(p_hflip)
        self.vflip = transforms.RandomVerticalFlip(p_vflip)
        self.to_tensor = transforms.PILToTensor()
        # self.brightness = transforms.ColorJitter(brightness=0.3)
        # self.space_to_depth = SpaceToDepth(args.k, args.rerange)

    def __call__(self, img):
        img = self.crop(img)
        img = self.hflip(img)
        img = self.vflip(img)
        img = self.to_tensor(img)
        img = img.to(torch.int32)
        # img = self.brightness(img)
        # img = self.space_to_depth(img)
        return img


class EvalTransform:
    def __init__(self, patch_sz):
        self.crop = transforms.CenterCrop(patch_sz)
        self.to_tensor = transforms.PILToTensor()
        # self.space_to_depth = SpaceToDepth(args.k, args.rerange)

    def __call__(self, img):
        img = self.crop(img)
        img = self.to_tensor(img)
        img = img.to(torch.int32)
        # img = self.space_to_depth(img)
        return img


def raw_to_tensor(img):
    """Convert a uint16 image to a float tensor scaled between 0 and 1."""
    img = torch.from_numpy(img)
    # img = img / 65535.0
    # HWC to CHW
    img = img.permute(2, 0, 1)
    img = img.to(torch.int32)

    return img


class RandomCropWithCond:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, cond):
        rect = transforms.RandomCrop.get_params(img, output_size=self.size)
        img = TF.crop(img, *rect)
        cond = TF.crop(cond, *rect)
        return img, cond


class RAWTrainTransform:
    def __init__(self, patch_sz):
        self.crop = RandomCropWithCond(patch_sz)
        self.raw_to_tensor = raw_to_tensor
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, cond):
        img = self.raw_to_tensor(img)
        cond = self.to_tensor(cond)
        img, cond = self.crop(img, cond)
        return img, cond


class RAWEvalTransform:
    def __init__(self, patch_sz):
        self.crop = transforms.CenterCrop(patch_sz)
        self.raw_to_tensor = raw_to_tensor
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, cond):
        img = self.raw_to_tensor(img)
        cond = self.to_tensor(cond)
        img = self.crop(img)
        cond = self.crop(cond)

        return img, cond
