import os 
from glob import glob
import numpy as np 
from collections import defaultdict
import pydicom
import PIL.Image
import h5py
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage import exposure


class MedicalImageDeal:
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)


def get_im_paths(root='.', data_root='LVSC_raw/', raw_img_path='CAP_challenge_training/'):
    """Fetch image paths, assuming the dir structure is 
       {data_root}/CAP_challenge_training/CAP_challenge_training_{number} and 
       {data_root}/CAP_challenge_training/CAP_challenge_training_masks
    """
    
    all_patients = glob(f"{root}/{data_root}{raw_img_path}*/*")
    
    training_img = defaultdict(list)
    img_count = 0
    patient_count = 0
    gt_count = 0
    for patient in all_patients:
        if "EDES" not in patient:
            patient_count += 1
            imgs = glob(f"{patient}/*.dcm")
            for img in imgs:
                img_count += 1
                try:
                    slice = img.split('/')[-1].split('.')[0]
                    p_id = patient.split('/')[-1]
                    cycle = slice.split('_')[1]
                    p_id_cycle = f'{p_id}_{cycle}'
                    gt = f"{root}/{data_root}CAP_challenge_training_masks/{p_id}/{slice}.png"
                    training_img[p_id_cycle].append((img, gt))
                    gt_count += 1
                except FileNotFoundError:
                    print(f'Ground truth for {img} not found.') 
    
    assert img_count == gt_count, "Number of images and ground truths do not match."
    
    print(f"Total number of patients: {patient_count}")
    print(f"Total number of slices: {img_count}")
    
    return training_img
    

def read_dicom(path):
    dcm = pydicom.dcmread(path)
    return dcm.pixel_array, dcm


def get_mask(path, discard_alpha=True):
    with PIL.Image.open(path) as img:
        mask = np.array(img)    
    
    # Discard the 4th channel (alpha) for png images
    if discard_alpha and mask.shape[-1] == 4:
        mask = mask[:, :, :3]
        
    # EDA shows that all channels are the same
    mask = np.mean(mask, axis=-1)
    
    # Normalize mask to 0 and 1
    if mask.max() == 255:
        mask = mask / 255
        
    return mask.astype(np.uint8) 
    
    
def resample_image(image, current_spacing, target_spacing=(1.45, 1.45)):
    """
    Resample image to the average resolution of 1.45mm^2 per paper recommendation
    https://arxiv.org/abs/2007.01152
    """
    resampling_factor = [current_spacing[i]/target_spacing[i] 
                         for i in range(len(current_spacing))]
    resampled_image = zoom(image, resampling_factor, order=1)  
    return resampled_image


def crop_or_pad(image, target_size=(224, 224)):
    current_size = image.shape

    # Ensure padding does not lead to negative sizes
    pad_width = [
        (
            max(0, (target_size[i] - current_size[i]) // 2), 
            max(0, target_size[i] - current_size[i] - (target_size[i] - current_size[i]) // 2)
        ) 
        for i in range(2)
    ]
    image_padded = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    current_padded_size = image_padded.shape
    
    # Calculate crop to ensure we do not end up with zero size
    crop = [
        (
            max(0, (current_padded_size[i] - target_size[i]) // 2),
            max(0, (current_padded_size[i] - target_size[i]) // 2 + target_size[i])
        )
        for i in range(2)
    ]
    image_cropped = image_padded[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    
    # Additional checks to prevent zero-size dimensions
    if image_cropped.size == 0:
        raise ValueError("Resulting cropped image has zero size.")
    
    return image_cropped


def normalize_image(image, median, iqr):
    """
    Normalize the images of each patient by removing the median and dividing 
    by the interquartile range computed on his MRI scan
    https://arxiv.org/abs/2007.01152
    """
    normalized_image = (image - median) / iqr
    return normalized_image


def generate_sort_key(paths):
    p = paths[0].split('/')[-1].split('.')[0].split('_')[-1]
    p = [i for i in p if i.isdigit()]
    return int(''.join(p))

    
def preprocessing(root='.', data_root='LVSC_raw/', raw_img_path='CAP_challenge_training/', 
                  output_dir="./LVSC_preprocessed_slices/"):
    imgs = get_im_paths(root=root, data_root=data_root, raw_img_path=raw_img_path)
    patients = list(imgs.keys())
    
    n_slices = 0
    for p in tqdm(patients):
        print(f"Processing patient {p}")
        slices = imgs[p]
        
        patient_mri_volume = np.stack([read_dicom(s[0])[0] for s in slices])
        median = np.median(patient_mri_volume)
        iqr = np.percentile(patient_mri_volume, 75) - np.percentile(patient_mri_volume, 25)
        
        slices = sorted(slices, key=generate_sort_key)
        slice_num = 0
        for s in slices:
            img, dcm = read_dicom(s[0])
            mask = get_mask(s[1])

            # Resample image to the average resolution of 1.45mm^2
            img = resample_image(img, dcm.PixelSpacing)
            
            img = crop_or_pad(img)
            img = MedicalImageDeal(img, percent=0.99).valid_img
            mask = crop_or_pad(mask)
            
            if img.shape != mask.shape:
                print(f"Image and mask shapes do not match for {s[0]}")
                continue
            
            # Normalize the image
            img = normalize_image(img, median, iqr)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to HDF5
            h5_path = os.path.join(output_dir, f"{p}_slice_{slice_num}.h5")
            with h5py.File(h5_path, 'w') as h5f:
                h5f.create_dataset('image', data=img, dtype='float32')
                h5f.create_dataset('label', data=mask, dtype='uint8')
                
            n_slices += 1
            slice_num += 1
                
    print(f"Total number of slices: {n_slices}")
            
    
if __name__ == "__main__":
    root='.'
    data_root='LVSC_raw/'
    raw_img_path='CAP_challenge_training/' 
    output_dir="./LVSC_preprocessed_slices/"
    
    preprocessing(root=root, data_root=data_root, raw_img_path=raw_img_path, 
                  output_dir=output_dir)
