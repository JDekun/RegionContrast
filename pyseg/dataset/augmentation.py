import random
import math
import numbers
import collections

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F


class Compose(object):
    """
    Composes several segsegtransforms together.

    Args:
        segtransforms (List[Transform]): list of segtransforms to compose.

    Example:
        segtransforms.Compose([
            segtransforms.CenterCrop(10),
            segtransforms.ToTensor()])
    """
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label):
        for t in self.segtransforms:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    # Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (1 x C x H x W).
    def __call__(self, image, label):
        if isinstance(image, Image.Image) and isinstance(label, Image.Image):
            image = np.asarray(image).copy()
            label = np.asarray(label).copy()
        elif not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                   "[eg: data readed by PIL.Image.open()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1))[np.newaxis])
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label[np.newaxis, np.newaxis])
        if not isinstance(label, torch.FloatTensor):
            label = label.float()
        return image, label


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
            self.std = torch.Tensor(np.float32(std)[:, np.newaxis, np.newaxis])
        self.mean = torch.Tensor(np.float32(mean)[:, np.newaxis, np.newaxis])

    def __call__(self, image, label):
        assert image.size(1) == len(self.mean)
        if self.std is None:
            image -= self.mean
        else:
            image -= self.mean
            image /= self.std
        return image, label


class Resize(object):
    """
    Resize the input tensor to the given size.
    'size' is a 2-element tuple or list in the order of (h, w)
    """
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        image = F.interpolate(image, size=self.size, mode='bilinear', align_corners=False)
        label = F.interpolate(label, size=self.size, mode='nearest')
        return image, label

class ResizeLongSize(object):
    """
    Resize the long size of the input image into fix size
    """
    def __init__(self, size=2048):
        assert type(size) == int , "Long size must be an integer"
        self.size = size

    def __call__(self, image, label):
        _,_,h,w = image.size()
        if h > w:
            w_r = int( self.size * w / h)
            image = F.interpolate(image, size=( self.size, w_r), mode='bilinear', align_corners=False)
            label = F.interpolate(label, size=( self.size, w_r), mode='nearest')
        else:
            h_r = int(2048 * h / w)
            image = F.interpolate(image, size=(h_r, self.size),mode='bilinear', align_corners=False)
            label = F.interpolate(label, size=(h_r, self.size), mode='nearest')

        return image, label

class RandResize(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    """
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number):
            self.scale = scale
        else:
            raise (RuntimeError("segtransforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransforms.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1. - self.scale[0]) * random.random()
        else:
            temp_scale = 1. + (self.scale[1] - 1.) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        h, w = image.size()[-2:]
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        label = F.interpolate(label, size=(new_h, new_w), mode='nearest')
        return image, label


class Crop(object):
    """Crops the given tensor.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if isinstance(ignore_label, int):
            ##!!!!!!change to 0
            self.ignore_label = 0
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = image.size()[-2:]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
            image = F.pad(image, border, mode='constant', value=0.)
            label = F.pad(label, border, mode='constant', value=self.ignore_label)
        h, w = image.size()[-2:]
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
        image = image[:, :, h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[:, :, h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label


class RandRotate(object):
    """
    Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    """
    def __init__(self, rotate, ignore_label=255):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number):
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransforms.RandRotate() scale param error.\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label

    def __call__(self, image, label):
        angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
        M = cv2.getRotationMatrix2D((0, 0), angle, 1)
        t_M = torch.Tensor(M).unsqueeze(dim=0)
        grid = F.affine_grid(t_M, image.size())

        image = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
        label += 1
        label = F.grid_sample(label, grid, mode='nearest', align_corners=False)
        label[label == 0.] = self.ignore_label + 1
        label -= 1
        return image, label


class RandomHorizontalFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = torch.flip(image, [3])
            label = torch.flip(label, [3])
        return image, label


class RandomVerticalFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = torch.flip(image, [2])
            label = torch.flip(label, [2])
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=2):
        self._filter = GaussianBlur(radius=radius)

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = self._filter(image)
        return image, label


class GaussianBlur(nn.Module):
    def __init__(self, radius):
        super(GaussianBlur, self).__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.sigma = 0.3 * (self.radius - 1) + 0.8
        self.kernel = nn.Conv2d(3, 3, self.kernel_size, stride=1,
                                padding=self.radius, bias=False, groups=3)
        self.weight_init()

    def forward(self, input):
        assert input.size(1) == 3
        return self.kernel(input)

    def weight_init(self):
        weights = np.zeros((self.kernel_size, self.kernel_size))
        weights[self.radius, self.radius] = 1
        weight = gaussian_filter(weights, sigma=self.sigma)
        for param in self.kernel.parameters():
            param.data.copy_(torch.from_numpy(weight))
            param.requires_grad = False
