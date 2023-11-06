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
    

def read_dicom(path):
    dcm = pydicom.dcmread(path)
    return dcm.pixel_array, dcm


def get_mask(path):
    with PIL.Image.open(path) as img:
        mask = np.array(img)          
    return mask.astype(np.uint8) 


def resample_image(image, current_spacing, target_spacing=(1.89, 1.89)):
    resampling_factor = [current_spacing[i]/target_spacing[i] 
                         for i in range(len(current_spacing))]
    resampled_image = zoom(image, resampling_factor, order=1)  
    return resampled_image


def crop_or_pad(image, target_size=(189, 189)):
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


def normalize_image(image, dcm):
    "By ChatGPT"
    image = image.astype(np.int16)
    
    # Define window level and window width
    WL = 40  # For liver
    WW = 200  # For liver

    # Set the window boundaries
    window_low = WL - WW // 2
    window_high = WL + WW // 2
    
    # Offset by the minimum HU value (-1024 for air)
    image[image == -2000] = -1024  # Replace any ignored values (e.g., for outside the scan) with air HU value

    # Convert to Hounsfield units (HU)
    intercept = np.int16(dcm.RescaleIntercept)
    slope = dcm.RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
    image += np.int16(intercept)

    # Apply windowing
    image_clipped = np.clip(image, window_low, window_high)
    
    # Normalize the image to [-1, 1]
    image_normalized = ((image_clipped - window_low) / (window_high - window_low)) * 2 - 1

    # Make sure everything is within the new range
    image_normalized = np.clip(image_normalized, -1, 1)
    
    return image_normalized
    
    
def get_im_ct_paths(root='.', data_root='raw_CHAOS_data'):
    patients = glob(f"{root}/{data_root}/Train_Sets/CT/*")
    patients = sorted(patients, key=lambda x: int(x.split('/')[-1]))
    
    training_img = defaultdict(list)
    slice_count = 0
    for p in patients:
        patient_id = p.split('/')[-1]
        try:
            im_paths = sorted(glob(f"{p}/*/*.dcm"), 
                            key=lambda x: int(x.split('/')[-1].split(',')[0].replace('i', '')))
        except ValueError:
            im_paths = sorted(glob(f"{p}/*/*.dcm"), 
                            key=lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
        gt_paths = sorted(glob(f"{p}/*/*.png"),
                          key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
                          
        assert len(im_paths) == len(gt_paths), "Number of images and ground truths do not match."
        
        for im_path, gt_path in zip(im_paths, gt_paths):
            training_img[patient_id].append((im_path, gt_path))
            slice_count += 1
    
    print(f"Total number of patients: {len(training_img)}")
    print(f"Total number of slices: {slice_count}")
    return training_img


def preprocessing(root='.', data_root='raw_CHAOS_data', output_dir='CHAOS_CT_preprocessed_slices/'):
    imgs = get_im_ct_paths(root=root, data_root=data_root)
    for p in tqdm(imgs.keys()):
        print('Processing patient {}'.format(p))
        slices = imgs[p]
        
        slice_count = 0
        for s in slices:
            img, dcm = read_dicom(s[0])
            mask = get_mask(s[1])
            
            # Resample image to the average resolution of 1.89mm^2
            img = resample_image(img, dcm.PixelSpacing)
        
            # Normalize the image
            img = normalize_image(img, dcm)
            
            # Crop or pad the image to 224x224
            img = crop_or_pad(img)
            img = MedicalImageDeal(img, percent=0.99).valid_img
            mask = crop_or_pad(mask)
            
            if img.shape != mask.shape:
                print(f"Image and mask shapes do not match for {s[0]}")
                continue

            os.makedirs(output_dir, exist_ok=True)
            
            # Save to HDF5
            h5_path = os.path.join(output_dir, f"patient_{p}_slice_{slice_count}.h5")
            with h5py.File(h5_path, 'w') as h5f:
                h5f.create_dataset('image', data=img, dtype='float32')
                h5f.create_dataset('label', data=mask, dtype='uint8')
                
            slice_count += 1


if __name__ == '__main__':
    preprocessing()