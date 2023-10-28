import os 
from glob import glob
import numpy as np 
import h5py
from tqdm import tqdm
import random 
import shutil


def train_val_test_split(search_path, split_ratios, seed=None):
    if sum(split_ratios) != 1:
        raise ValueError('Sum of split ratios must be 1.')
    
    if seed is not None:
        random.seed(seed)
    
    slices = glob(f'{search_path}*.h5')
    patients = list(set([slice.split('/')[-1].split('_')[0] for slice in slices]))
    print('Number of patients:', len(patients))
    print('Number of slices:', len(slices))
    
    random.shuffle(patients)
    
    train_idx = int(len(patients) * split_ratios[0])
    val_idx = train_idx + int(len(patients) * split_ratios[1])

    train_patients = patients[:train_idx]
    val_patients = patients[train_idx:val_idx]
    test_patients = patients[val_idx:]
    
    assert len(train_patients) + len(val_patients) + len(test_patients) == len(patients)

    train_slices = [slice for slice in slices if slice.split('/')[-1].split('_')[0] in train_patients]
    val_slices = [slice for slice in slices if slice.split('/')[-1].split('_')[0] in val_patients]
    test_slices = [slice for slice in slices if slice.split('/')[-1].split('_')[0] in test_patients]
    
    print('Number of train slices:', len(train_slices))
    print('Number of val slices:', len(val_slices))
    print('Number of test slices:', len(test_slices))
          
    assert len(train_slices) + len(val_slices) + len(test_slices) == len(slices)
    
    return (train_patients, train_slices), (val_patients, val_slices), (test_patients, test_slices)


def save_volume_h5(patients, slices, txt_file_path, h5_dir, task='val'):
    print(f'Copying {task} volumes')
    for p in patients:
        slices_ = [slice for slice in slices if p in slice]
        slice_phase = set(['_'.join(slice.split('/')[-1].split('_')[:2]) for slice in slices_])
        for i in slice_phase:
            volume = [slice for slice in slices_ if i in slice]
            volume = sorted(volume, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            images = []
            labels = []
            for v in volume:
                with h5py.File(v, 'r') as h5f:
                    images.append(h5f['image'][:])
                    labels.append(h5f['label'][:])
            images = np.stack(images)
            labels = np.stack(labels)  
            
            with open(txt_file_path, 'a') as f:
                f.write(f"{h5_dir.split('/')[-1]}/{i}.h5" + "\n")

            h5_path = os.path.join(h5_dir, f"{i}.h5")
            with h5py.File(h5_path, 'w') as h5f:
                h5f.create_dataset('image', data=images, dtype='float32')
                h5f.create_dataset('label', data=labels, dtype='uint8')

def generate_train_val_split_sets(root='.', search_path='LVSC_preprocessed_slices/', split_ratios=(0.75, 0.1, 0.15),
                                  seed=10):
    search_path = f'{root}/{search_path}'
    
    (_, train_slices), (val_patients, val_slices), (test_patients, test_slices) = train_val_test_split(search_path, split_ratios, seed)
    
    ####### Train val test split
    # Train data are split into slices
    # Val and test data are split into volumes

    # dir structures
    # LVSC/data/
    #   * val.txt 
    #   * train.txt 
    #   * test.txt

    # LVSC_data/
    #   * training_slices/ 
    #   * val_volumes/ 
    #   * testing_volumes/
    
    os.makedirs(f'{root}/LVSC', exist_ok=True)
    os.makedirs(f'{root}/LVSC/data', exist_ok=True)
    os.makedirs(f'{root}/LVSC_data', exist_ok=True)
    os.makedirs(f'{root}/LVSC_data/training_slices', exist_ok=True)
    os.makedirs(f'{root}/LVSC_data/val_volumes', exist_ok=True)
    os.makedirs(f'{root}/LVSC_data/testing_volumes', exist_ok=True)

    with open(f'{root}/LVSC/data/train.txt', 'w') as f:
        for slice in train_slices:
            f.write("training_slices/" + slice.split('/')[-1] + '\n')
    
    # Copy training slices
    print('Copying training slices')
    for slice in tqdm(train_slices):
        destination = f'{root}/LVSC_data/training_slices/' + slice.split('/')[-1]
        shutil.copy(slice, destination)
    
    # Val and test sets need volume level data
    save_volume_h5(val_patients, val_slices, f'{root}/LVSC/data/val.txt', f'{root}/LVSC_data/val_volumes')
    save_volume_h5(test_patients, test_slices, f'{root}/LVSC/data/test.txt', f'{root}/LVSC_data/testing_volumes',
                   task='test')
    

if __name__ == "__main__":
    generate_train_val_split_sets()