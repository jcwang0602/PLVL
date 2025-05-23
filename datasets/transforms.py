import random

import numpy as np
import torch
import torch.nn.functional as FF
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageEnhance, ImageFilter

from utils.box_utils import xyxy2xywh


def crop(image, box, gt_mask, region):
    cropped_image = F.crop(image, *region)
    if gt_mask is not None:
        gt_mask = F.crop(gt_mask, *region)
    i, j, h, w = region
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    return cropped_image, cropped_box, gt_mask


def resize_according_to_long_side(img, box, gt_mask, size):
    h, w = img.height, img.width
    ratio = float(size / float(max(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio
    if gt_mask is not None:
        gt_mask = F.resize(gt_mask, (new_h, new_w))

    return img, box, gt_mask


def resize_according_to_short_side(img, box, gt_mask, size):
    h, w = img.height, img.width
    ratio = float(size / float(min(h, w)))
    new_w, new_h = round(w * ratio), round(h * ratio)
    img = F.resize(img, (new_h, new_w))
    box = box * ratio
    if gt_mask is not None:
        gt_mask = F.resize(gt_mask, (new_h, new_w))
    return img, box, gt_mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict):
        for t in self.transforms:  # [500,375]->[448,336]->[448,336]->[448,336]->[3,336,448]
            input_dict = t(input_dict)
        return input_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, input_dict):
        if random.random() < 0.8:
            image = input_dict["img"]
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    image = self.rand_brightness(image)
                elif func_id == 1:
                    image = self.rand_contrast(image)
                elif func_id == 2:
                    image = self.rand_saturation(image)
            input_dict["img"] = image

        return input_dict


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.0

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict["img"]
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict["img"] = img

        return input_dict


class RandomHorizontalFlip(object):
    def __call__(self, input_dict):
        if random.random() < 0.5:
            img = input_dict["img"]
            box = input_dict["box"]
            text = input_dict["text"]
            gt_mask = None
            if "gt_mask" in input_dict:
                gt_mask = input_dict["gt_mask"]

            img = F.hflip(img)
            if gt_mask is not None:
                gt_mask = F.hflip(gt_mask)
            text = text.replace("right", "*&^special^&*").replace("left", "right").replace("*&^special^&*", "left")
            h, w = img.height, img.width
            box = box[[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

            input_dict["img"] = img
            input_dict["box"] = box
            input_dict["text"] = text
            if gt_mask is not None:
                input_dict["gt_mask"] = gt_mask

        return input_dict


class RandomResize(object):
    def __init__(self, sizes, with_long_side=True):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.with_long_side = with_long_side

    def __call__(self, input_dict):
        img = input_dict["img"]
        box = input_dict["box"]
        gt_mask = None
        if "gt_mask" in input_dict:
            gt_mask = input_dict["gt_mask"]
        size = random.choice(self.sizes)
        if self.with_long_side:
            resized_img, resized_box, resized_gt_mask = resize_according_to_long_side(img, box, gt_mask, size)
        else:
            resized_img, resized_box, resized_gt_mask = resize_according_to_short_side(img, box, gt_mask, size)

        input_dict["img"] = resized_img
        input_dict["box"] = resized_box
        if gt_mask is not None:
            input_dict["gt_mask"] = resized_gt_mask
        return input_dict


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, input_dict):
        img = input_dict["img"]
        box = input_dict["box"]
        gt_mask = None
        if "gt_mask" in input_dict:
            gt_mask = input_dict["gt_mask"]

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])  # [i, j, target_w, target_h]
            box_xywh = xyxy2xywh(box)
            box_x, box_y = box_xywh[0], box_xywh[1]
            if box_x > region[0] and box_y > region[1]:
                img, box, gt_mask = crop(img, box, gt_mask, region)
                input_dict["img"] = img
                input_dict["box"] = box
                if gt_mask is not None:
                    input_dict["gt_mask"] = gt_mask
                return input_dict

        return input_dict


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict):
        text = input_dict["text"]

        dir_words = ["left", "right", "top", "bottom", "middle"]
        for wd in dir_words:
            if wd in text:
                return self.transforms1(input_dict)

        if random.random() < self.p:
            return self.transforms2(input_dict)
        else:
            return self.transforms1(input_dict)


class ToTensor(object):
    def __call__(self, input_dict):
        img = input_dict["img"]
        img = F.to_tensor(img)
        if "gt_mask" in input_dict:
            gt_mask = input_dict["gt_mask"]
            gt_mask = torch.from_numpy(np.array(gt_mask))
            gt_mask[gt_mask > 0.001] = 1.0
            input_dict["gt_mask"] = gt_mask
        input_dict["img"] = img

        return input_dict


class NormalizeAndPad(object):
    def __init__(self, model_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict):
        img = input_dict["img"]  # [3, 336, 448]
        if "box" in input_dict:
            box = input_dict["box"]
            _, h, w = img.shape
            box[0], box[2] = (box[0] / w) * self.size, (box[2] / w) * self.size
            box[1], box[3] = (box[1] / h) * self.size, (box[3] / h) * self.size
            input_dict["box"] = box
        img = FF.interpolate(img.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=True).squeeze(0)
        if "gt_mask" in input_dict:
            gt_mask = input_dict["gt_mask"]
            gt_mask = FF.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(), size=(self.size, self.size), mode="nearest").squeeze(0).squeeze(0)
            input_dict["gt_mask"] = gt_mask
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        # dw = (self.size - w) / 2.0
        # dh = (self.size - h) / 2.0
        # top, bottom = round(dh - 0.1), round(dh + 0.1)
        # left, right = round(dw - 0.1), round(dw + 0.1)

        out_img = torch.zeros((3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()
        out_img[:, top : top + h, left : left + w] = img
        out_mask[top : top + h, left : left + w] = 0

        input_dict["img"] = out_img
        input_dict["mask"] = out_mask

        if "gt_mask" in input_dict:
            gt_mask = input_dict["gt_mask"]
            out_gt_mask = torch.zeros((self.size, self.size))
            out_gt_mask[top : top + h, left : left + w] = gt_mask
            input_dict["gt_mask"] = out_gt_mask

        if "box" in input_dict.keys():
            box = input_dict["box"]
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            h, w = out_img.shape[-2:]
            box = xyxy2xywh(box)
            box = box / torch.tensor([w, h, w, h], dtype=torch.float32)
            input_dict["box"] = box

        return input_dict
