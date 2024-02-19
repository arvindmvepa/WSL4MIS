# https://github.com/BWGZK/CycleMix/blob/main/data/mscmr.py

import os 
from pathlib import Path
from torch.utils import data
import nibabel as nib
import numpy as np 
import mscmr_transforms as T
import h5py
    

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def read_image(img_path):
    img_dat = load_nii(img_path)
    img = img_dat[0]
    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
    target_resolution = (1.36719, 1.36719)
    scale_vector = (pixel_size[0] / target_resolution[0],
                    pixel_size[1] / target_resolution[1])
    img = img.astype(np.float32)
    return [(img-img.mean())/img.std(), scale_vector]
    
    
def read_label(lab_path):
    lab_dat = load_nii(lab_path)
    lab = lab_dat[0]
    pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
    target_resolution = (1.36719, 1.36719)
    scale_vector = (pixel_size[0] / target_resolution[0],
                    pixel_size[1] / target_resolution[1])
    # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
    return [lab, scale_vector]
    
    
def make_transforms():
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    return T.Compose([
        T.Rescale(),
        T.PadOrCropToSize([212,212]),
        normalize,
    ])


def mscmr_processing_train(root, transform=None, out_dir='./OUTPUT/MSCMR'):
    
    img_folder = Path(root + '/' + 'train' + '/images')
    lab_folder = Path(root + '/' + 'train' + '/labels')
        
    img_paths = sorted(list(img_folder.iterdir()))
    lab_paths = sorted(list(lab_folder.iterdir()))
    
    examples = []
    img_dict = {}
    lab_dict = {}
    
    for img_path, lab_path in zip(img_paths, lab_paths):
        img = read_image(str(img_path))
        img_name = img_path.stem
        img_dict.update({img_name : img})
        lab = read_label(str(lab_path))
        lab_name = lab_path.stem
        lab_dict.update({lab_name : lab})
        print(img_name, lab_name)
        
        assert img[0].shape[2] == lab[0].shape[2]
        examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/' + 'training_slices', exist_ok=True)
    os.makedirs(out_dir + '_txt' + '/', exist_ok=True)
    
    for idx in range(len(examples)):
        img_name, lab_name, Z, X, Y = examples[idx]
        if Z != -1:
            img = img_dict[img_name][Z, :, :]
            lab = lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = img_dict[img_name][:, X, :]
            lab = lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = img_dict[img_name][0][:, :, Y]
            scale_vector_img = img_dict[img_name][1]
            lab = lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
        if transform:
            img, target = transform([img, scale_vector_img], [target,scale_vector_lab])
        img = img.detach().numpy()[0]
        label = target['masks'].detach().numpy()[0]
        pt_id = img_name.split('_')[0][7:]
        h5_path = os.path.join(out_dir + '/' + 'training_slices', f"patient_{pt_id}_slice_{Y}.h5")
        print(h5_path)
        
        with open(out_dir + '_txt' + '/' + 'train.txt', 'a') as f:
            f.write('training_slices/' + f"patient_{pt_id}_slice_{Y}.h5" + '\n')
        
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('image', data=img, dtype='float32')
            h5f.create_dataset('scribble', data=label, dtype='uint8')
            

def mscmr_processing_test(root, img_task='test', out_dir='./OUTPUT/MSCMR', transform=None):
    """img_task: 'test' or 'val'"""
    if img_task == 'test':
        img_folder = Path(root + '/' + 'TestSet' + '/images')
        lab_folder = Path(root + '/' + 'TestSet' + '/labels')
        save_dir = 'testing_volumes'
    elif img_task == 'val':
        img_folder = Path(root + '/' + 'val' + '/images')
        lab_folder = Path(root + '/' + 'val' + '/labels')
        save_dir = 'val_volumes'
        
    img_paths = sorted(list(img_folder.iterdir()))
    lab_paths = sorted(list(lab_folder.iterdir()))
    
    examples = {}
    img_dict = {}
    lab_dict = {}
    
    for img_path, lab_path in zip(img_paths, lab_paths):
        img = read_image(str(img_path))
        img_name = img_path.stem
        img_dict.update({img_name : img})
        lab = read_label(str(lab_path))
        lab_name = lab_path.stem
        lab_dict.update({lab_name : lab})
        print(img_name, lab_name)
        
        assert img[0].shape[2] == lab[0].shape[2]
        examples[img_name] = [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/' + save_dir, exist_ok=True)
    os.makedirs(out_dir + '_txt' + '/', exist_ok=True)
    
    for p, slices in examples.items():
        stacked_img_arr = []
        stacked_label_arr = []
        for idx in range(len(slices)):
            img_name, lab_name, Z, X, Y = slices[idx]
            if Z != -1:
                img = img_dict[img_name][Z, :, :]
                lab = lab_dict[lab_name][Z, :, :]
            elif X != -1:
                img = img_dict[img_name][:, X, :]
                lab = lab_dict[lab_name][:, X, :]
            elif Y != -1:
                img = img_dict[img_name][0][:, :, Y]
                scale_vector_img = img_dict[img_name][1]
                lab = lab_dict[lab_name][0][:, :, Y]
                scale_vector_lab = lab_dict[lab_name][1]
            else:
                raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
            img = np.expand_dims(img, 0)
            lab = np.expand_dims(lab, 0)
            target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
            if transform:
                img, target = transform([img, scale_vector_img], [target,scale_vector_lab])
            img = img.detach().numpy()
            label = target['masks'].detach().numpy()
            stacked_img_arr.append(img)
            stacked_label_arr.append(label)
        img = np.vstack(stacked_img_arr)
        label = np.vstack(stacked_label_arr)
        pt_id = img_name.split('_')[0][7:]
        h5_path = os.path.join(out_dir + '/' + save_dir, f"patient_{pt_id}.h5")
            
        with open(out_dir + '_txt' + '/' + f'{img_task}.txt', 'a') as f:
            f.write(f'{save_dir}/' + f"patient_{pt_id}.h5" + '\n')    
        
        print(h5_path)
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('image', data=img, dtype='float32')
            h5f.create_dataset('label', data=label, dtype='uint8')


if __name__=="__main__":
    root = "./MSCMR_dataset"
    out_dir = "./OUTPUT/MSCMR"
    
    mscmr_processing_train(root, transform=make_transforms(),
                     out_dir='./OUTPUT/MSCMR') 
    
    mscmr_processing_test(root, img_task='test', transform=make_transforms(),
                         out_dir='./OUTPUT/MSCMR') 
    
    mscmr_processing_test(root, img_task='val', transform=make_transforms(),
                         out_dir='./OUTPUT/MSCMR') 
    