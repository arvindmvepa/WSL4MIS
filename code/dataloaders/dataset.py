import itertools
import os
import random
import h5py
import numpy as np
import torch
import cv2
import math
import sys
from scipy import ndimage
from PIL import Image
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from skimage.morphology import skeletonize, dilation, closing


def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    img = Image.fromarray(image)
    img_rotate = img.rotate(angle)
    return img_rotate


def translate_img(img, x_shift, y_shift):

    (height, width) = img.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_largest_two_component_2D(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(2, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = [img]
    else:
        if(threshold):
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            component1 = labeled_array == max_label1
            out_img = [component1]
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab[0]
                    out_img.append(temp_cmp)
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            if max_label2.shape[0] > 1:
                max_label2 = max_label2[0]
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                out_img = [component1, component2]
            else:
                out_img = [component1]
    return out_img


class Cutting_branch(object):
    def __init__(self):
        self.lst_bifur_pt = 0
        self.branch_state = 0
        self.lst_branch_state = 0
        self.direction2delta = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [
            0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1], 7: [1, 0], 8: [1, 1]}

    def __find_start(self, lab):
        y, x = lab.shape
        idxes = np.asarray(np.nonzero(lab))
        for i in range(idxes.shape[1]):
            pt = tuple([idxes[0, i], idxes[1, i]])
            assert lab[pt] == 1
            directions = []
            for d in range(9):
                if d == 4:
                    continue
                if self.__detect_pt_bifur_state(lab, pt, d):
                    directions.append(d)
            if len(directions) == 1:
                start = pt
                self.start = start
                self.output[start] = 1
                return start
        start = tuple([idxes[0, 0], idxes[1, 0]])
        self.output[start] = 1
        self.start = start
        return start

    def __detect_pt_bifur_state(self, lab, pt, direction):

        d = direction
        y = pt[0] + self.direction2delta[d][0]
        x = pt[1] + self.direction2delta[d][1]
        if lab[y, x] > 0:
            return True
        else:
            return False

    def __detect_neighbor_bifur_state(self, lab, pt):
        directions = []
        for i in range(9):
            if i == 4:
                continue
            if self.output[tuple([pt[0] + self.direction2delta[i][0], pt[1] + self.direction2delta[i][1]])] > 0:
                continue
            if self.__detect_pt_bifur_state(lab, pt, i):
                directions.append(i)

        if len(directions) == 0:
            self.end = pt
            return False
        else:
            direction = random.sample(directions, 1)[0]
            next_pt = tuple([pt[0] + self.direction2delta[direction]
                            [0], pt[1] + self.direction2delta[direction][1]])
            if len(directions) > 1 and pt != self.start:
                self.lst_output = self.output*1
                self.previous_bifurPts.append(pt)
            self.output[next_pt] = 1
            pt = next_pt
            self.__detect_neighbor_bifur_state(lab, pt)

    def __detect_loop_branch(self, end):
        for d in range(9):
            if d == 4:
                continue
            y = end[0] + self.direction2delta[d][0]
            x = end[1] + self.direction2delta[d][1]
            if (y, x) in self.previous_bifurPts:
                self.output = self.lst_output * 1
                return True

    def __call__(self, lab, seg_lab, iterations=1):
        self.previous_bifurPts = []
        self.output = np.zeros_like(lab)
        self.lst_output = np.zeros_like(lab)
        components = get_largest_two_component_2D(lab, threshold=15)
        if len(components) > 1:
            for c in components:
                start = self.__find_start(c)
                self.__detect_neighbor_bifur_state(c, start)
        else:
            c = components[0]
            start = self.__find_start(c)
            self.__detect_neighbor_bifur_state(c, start)
        self.__detect_loop_branch(self.end)
        struct = ndimage.generate_binary_structure(2, 2)
        output = ndimage.morphology.binary_dilation(
            self.output, structure=struct, iterations=iterations)
        shift_y = random.randint(-6, 6)
        shift_x = random.randint(-6, 6)
        if np.sum(seg_lab) > 1000:
            output = translate_img(output.astype(np.uint8), shift_x, shift_y)
            output = random_rotation(output)
        output = output * seg_lab
        return output


def scrible_2d(label, iteration=[4, 10]):
    lab = label
    skeleton_map = np.zeros_like(lab, dtype=np.int32)
    if np.sum(lab) == 0:
        return skeleton_map
    print("scrible_2d, debug1")
    sys.stdout.flush()
    struct = ndimage.generate_binary_structure(2, 2)
    if np.sum(lab) > 900 and iteration != 0 and iteration != [0] and iteration != None:
        iter_num = math.ceil(
            iteration[0]+random.random() * (iteration[1]-iteration[0]))
        slic = ndimage.morphology.binary_erosion(
            lab, structure=struct, iterations=iter_num)
    else:
        slic = lab
    print("scrible_2d, debug2")
    sys.stdout.flush()
    print("type(slic): {}".format(type(slic)))
    sys.stdout.flush()
    print("np.unique(slic): {}".format(np.unique(slic)))
    sys.stdout.flush()
    try:
        sk_slice = skeletonize(slic, method='lee')
    except Exception as e:
        print(e)
        raise
    print("scrible_2d, debug3")
    sys.stdout.flush()
    sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
    print("scrible_2d, debug4")
    sys.stdout.flush()
    skeleton_map = sk_slice
    print("scrible_2d, debug5")
    sys.stdout.flush()
    return skeleton_map


def scribble4class(label, class_id, class_num, iteration=[4, 10], cut_branch=True):
    print("Generating scribble for class {}".format(class_id))
    sys.stdout.flush()
    label = (label == class_id)
    sk_map = scrible_2d(label, iteration=iteration)
    print("sk_map.shape: {}, sk_map sum: {}".format(sk_map.shape, np.sum(sk_map)))
    sys.stdout.flush()
    if cut_branch and class_id != 0:
        cut = Cutting_branch()
        lab = sk_map
        if not (lab.sum() < 1):
            sk_map = cut(lab, seg_lab=label)
        print("after cut, sk_map.shape: {}, sk_map sum: {}".format(sk_map.shape, np.sum(sk_map)))
        sys.stdout.flush()
    if class_id == 0:
        class_id = class_num
    return sk_map * class_id


def generate_cutting_scribble(label, cut_branch=True):
    class_num = np.max(label) + 1
    output = np.zeros_like(label, dtype=np.uint8)
    print("Generating scribble for cutting branch")
    print("output.shape: {}".format(output.shape))
    sys.stdout.flush()
    for i in range(class_num):
        scribble = scribble4class(
            label, i, class_num, cut_branch=cut_branch)
        output += scribble.astype(np.uint8)
    print("Complete scribble generation for image")
    return output


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
                    elif self.scribble_gen == "cutting":
                        label = generate_cutting_scribble(h5f["label"][:])
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