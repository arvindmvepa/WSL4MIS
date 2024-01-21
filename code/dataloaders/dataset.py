import itertools
import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from skimage.morphology import skeletonize, dilation, closing


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


def generate_skeleton_scribble(mask):
    """ Scribbles are approximated by a skeleton of the image (only works for binary masks)
    :param mask: multi-channel binary mask
    :return: scribbles
    """
    # initialize scribbles as empty array
    scribbles = np.zeros_like(mask)
    assert len(mask.shape) == 2 and len(np.unique(mask)) <= 2, "only works for binary masks"
    m = np.copy(mask[:, :])
    skl = skeletonize(m)

    # make slightly thicker (but always inside the gt mask)
    skl = closing(skl)
    skl = dilation(skl) * m

    # assign skeleton to return array
    scribbles[...] = skl

    return scribbles


def generate_countor_scribble(mask, epsilon=1.0):
    """ Scribbles are approximated by a counter of the image (only works for binary masks)
    :param mask: multi-channel binary mask
    :return: scribbles
    """
    assert len(mask.shape) == 2 and len(np.unique(mask)) <= 2, "only works for binary masks"
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Simplify contours
    simplified_contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]

    # Create an empty image to draw simplified contours
    scribble_img = np.zeros_like(mask)

    # Draw the simplified contours
    scribble_img = cv2.drawContours(scribble_img, simplified_contours, -1, (255, 255, 255), 1)
    binary_scribble_img = scribble_img // 255

    # make slightly thicker (but always inside the gt mask)
    binary_scribble_img = closing(binary_scribble_img)
    binary_scribble_img = dilation(binary_scribble_img) * mask

    return binary_scribble_img


class BaseDataSets(Dataset):
    def __init__(self,  split='train', transform=None, sup_type="label", train_file="train.txt", val_file="val.txt",
                 data_root=".", scribble_gen=None):
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.scribble_gen = scribble_gen
        if self.scribble_gen:
            print(f"Using scribble_gen: {self.scribble_gen}")
        self.transform = transform
        if self.split == 'train':
            with open(train_file) as f:
                self.all_slices = f.read().splitlines()
            self.sample_list = self.all_slices

        elif self.split == 'val':
            with open(val_file) as f:
                self.all_volumes = f.read().splitlines()
            self.sample_list = self.all_volumes

        self.sample_list = [os.path.join(data_root, im_path) for im_path in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        with h5py.File(case, 'r') as h5f:  # Using 'with' ensures the file is automatically closed after the block
            if self.split == "train":
                image = h5f['image'][:]
                if self.scribble_gen:
                    if self.scribble_gen == "skeleton":
                        label = generate_skeleton_scribble(h5f["label"][:])
                    elif self.scribble_gen == "contour":
                        label = generate_countor_scribble(h5f["label"][:])
                    else:
                        raise ValueError(f"Value of self.scribble_gen: {self.scribble_gen} is not supported")
                else:
                    label = h5f[self.sup_type][:]
                sample = {'image': image, 'label': label}
                sample = self.transform(sample)
            else:
                image = h5f['image'][:]
                label = h5f['label'][:]
                sample = {'image': image, 'label': label}
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)