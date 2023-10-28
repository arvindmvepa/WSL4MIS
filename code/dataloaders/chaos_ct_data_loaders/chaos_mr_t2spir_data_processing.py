# save images in slice level
import glob
import os
from tqdm import tqdm 
import h5py
import numpy as np
import shutil
import SimpleITK as sitk

from scribbles_generator import generate_scribble
from utils import MedicalImageDeal
    

SEARCH_PATH = "./chaos_mri_data/*"
DATA_TYPE = "T2SPIR"
SLICES_WRITE_TO = f"./chaos_mri_{DATA_TYPE}_slices"
CLEAN_UP = False
n_test_patients = 4 
n_val_patients = 2

# saving images in slice level
def main():
    slice_num = 0
    for i in tqdm(sorted(glob.glob(SEARCH_PATH), key=lambda x: int(x.split('/')[-1]))):
        p = i + f"/{DATA_TYPE}/*"
        dirs = glob.glob(p)
        
        images = []
        labels = []
        scribbles = []
        print(dirs)
        for dir_ in dirs:
            if "DICOM_anon" in dir_:
                im_files = glob.glob(dir_ + "/*.npy")
                im_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
                for im_file in im_files:
                    image = np.load(im_file)
                    images.append(image[np.newaxis, :])
                    
            elif "Ground" in dir_:
                im_files = glob.glob(dir_ + "/*.npy")
                im_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
                for im_file in im_files:
                    label = np.load(im_file)
                    labels.append(label[np.newaxis, :])
                    
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        if images.shape != labels.shape:
                print("Error") 
        
        num_classes = np.unique(labels).size
        scribbles = generate_scribble(labels, tuple([1, num_classes-1]))
        # scribbles_classes = np.unique(scribbles)
        scribbles[scribbles == 0] = 255
        scribbles[scribbles == num_classes] = 0
        scribbles[scribbles == 255] = num_classes
        
        os.makedirs(SLICES_WRITE_TO, exist_ok=True)
        
        for slice_ind in range(images.shape[0]):                
            img = MedicalImageDeal(images[slice_ind], percent=0.99).valid_img
            img = (img - img.min()) / (img.max() - img.min())
            img = img.astype(np.float32)
            
            patient_num = int(i.split('/')[-1])
            item = f"{patient_num:0{2}d}"
            
            f = h5py.File(
                f'{SLICES_WRITE_TO}/patient{item}_slice_{slice_ind}.h5', 'w')
            f.create_dataset(
                'image', data=img, compression="gzip")
            f.create_dataset('label', data=labels[slice_ind], compression="gzip")
            f.create_dataset(
                'scribble', data=scribbles[slice_ind], compression="gzip")
            f.close()
            slice_num += 1

    print("Converted all chaos volumes to 2D slices")
    print("Total {} slices".format(slice_num))  
    
    ####### Train val test split
    # Train data are split into slices
    # Val and test data are split into volumes

    # dir structures
    # chaos/data/mri/
    #   * val.txt 
    #   * train.txt 
    #   * test.txt

    # chaos_data/mri/
    #   * training_slices/ 
    #   * val_slices/ 
    #   * testing_slices/
    
    print("Starting train-val-test split!!! ")
    search_dir = SLICES_WRITE_TO + '/*'

    all_slices = list(glob.glob(search_dir))
    print(all_slices[:2])

    all_patients = set([i.split('/')[-1].split('.')[0].split('_')[0] for i in all_slices])
    
    selected_patients = np.random.choice(list(all_patients), 
                                         n_test_patients + n_val_patients, replace=False)

    val_patients, test_patients = selected_patients[:n_val_patients], selected_patients[n_val_patients:]

    print("Selected patients for val set:", val_patients)
    print("Selected patients for test set:", test_patients)

    os.makedirs('./chaos', exist_ok=True)
    os.makedirs('./chaos/data', exist_ok=True)
    os.makedirs('./chaos/data/mri', exist_ok=True)
    os.makedirs('./chaos_data', exist_ok=True)        
    os.makedirs('./chaos/data', exist_ok=True)
    os.makedirs('./chaos_data/mri', exist_ok=True)
    os.makedirs('./chaos_data/mri/training_slices', exist_ok=True)
    os.makedirs('./chaos_data/mri/val_volumes', exist_ok=True)
    os.makedirs('./chaos_data/mri/testing_volumes', exist_ok=True)

    with open('./chaos/data/mri/train.txt', 'w') as f:
        for p in all_patients:
            if p not in val_patients and p not in test_patients:
                for s in all_slices:
                    if p in s:
                        f.write("training_slices/" + s.split('/')[-1] + '\n')

    # Writing training slices file names to train.txt              
    with open('./chaos/data/mri/train.txt', 'r') as f:
        train_slices = f.readlines()
        train_slices = [i.strip() for i in train_slices]

    for s in train_slices:
        file = f'{SLICES_WRITE_TO}/' + s.split('/')[-1]
        destination = './chaos_data/mri/' + s
        shutil.copy(file, destination)

    # Writing val and test volume files names to val.txt and test.txt
    for p in val_patients:
        print(p)
        slices = []
        volume_imgs = []
        volume_labels = []
        for s in all_slices:
            if p in s:
                slices.append(s)
        slices = sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for slice in slices: 
            file = h5py.File(slice, 'r')
            image = file['image'][:]
            label = file['label'][:]
            volume_imgs.append(image[np.newaxis, ...])
            volume_labels.append(label[np.newaxis, ...])
        
        volume_imgs = np.concatenate(volume_imgs, axis=0)
        volume_labels = np.concatenate(volume_labels, axis=0)
        f = h5py.File(
                './chaos_data/mri/val_volumes/patient_{}_volume.h5'.format(p[-2:]), 'w')
        f.create_dataset('image', data=volume_imgs, compression="gzip")
        f.create_dataset('label', data=volume_labels, compression="gzip")
        f.close()
        
        with open('./chaos/data/mri/val.txt', 'a') as f:
            f.write('val_volumes/patient_{}_volume.h5'.format(p[-2:]) + '\n')
        
    for p in test_patients:
        print(p)
        slices = []
        volume_imgs = []
        volume_labels = []
        for s in all_slices:
            if p in s:
                slices.append(s)
        slices = sorted(slices, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for slice in slices: 
            file = h5py.File(slice, 'r')
            image = file['image'][:]
            label = file['label'][:]
            volume_imgs.append(image[np.newaxis, ...])
            volume_labels.append(label[np.newaxis, ...])
        
        volume_imgs = np.concatenate(volume_imgs, axis=0)
        volume_labels = np.concatenate(volume_labels, axis=0)
        f = h5py.File(
                './chaos_data/mri/testing_volumes/patient_{}_volume.h5'.format(p[-2:]), 'w')
        f.create_dataset('image', data=volume_imgs, compression="gzip")
        f.create_dataset('label', data=volume_labels, compression="gzip")
        f.close()
        
        with open('./chaos/data/mri/test.txt', 'a') as f:
            f.write('testing_volumes/patient_{}_volume.h5'.format(p[-2:]) + '\n')
    
main()

# Clean up data
if CLEAN_UP:
    if os.path.exists(SLICES_WRITE_TO) and os.path.isdir(SLICES_WRITE_TO):
        print(f'Removing {SLICES_WRITE_TO}')
        shutil.rmtree(SLICES_WRITE_TO)