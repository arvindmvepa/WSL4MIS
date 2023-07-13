import glob

import h5py
import numpy as np
import SimpleITK as sitk
import re

# saving images in volume level


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


slice_num = 0
mask_path = sorted(
    glob.glob("./wsl4mis_data/ACDC/ACDC_test_raw/*_gt.nii.gz"))
for case in mask_path:
    label_itk = sitk.ReadImage(case)
    label = sitk.GetArrayFromImage(label_itk)

    image_path = case.replace("_gt", "")
    image_itk = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_itk)

    image = MedicalImageDeal(image, percent=0.99).valid_img
    image = (image - image.min()) / (image.max() - image.min())
    print(image.shape)
    image = image.astype(np.float32)
    item = re.split(r'/|\\', case)[-1].split(".")[0].replace("_gt", "")
    if image.shape != label.shape:
        print("Error")
    print(item)
    f = h5py.File(
        './wsl4mis_data/ACDC/ACDC_testing_volumes/{}.h5'.format(item), 'w')
    f.create_dataset(
        'image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()
    slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
