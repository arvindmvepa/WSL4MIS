import os 
from glob import glob 
from PIL import Image
import numpy as np
import h5py


root_dir = './DATA/DAVIS_2017'
trainval_reference_dir = './DATA/DAVIS_2016/ImageSets/480p'
test_reference_dir = './DATA/DAVIS_2017/DAVIS_test/ImageSets/2017'
output_root_dir = './OUTPUTS'


def load_jpg(image_path):
    # Load the image
    image = Image.open(image_path)
    return np.array(image)

# DATA/DAVIS_2017/JPEGImages/480p/lucia/00011.jpg
# DATA/DAVIS_2017/Annotations/480p/lucia/00011.png
def get_train_reference(train_path='train.txt'):
    path = trainval_reference_dir + '/' + train_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip().split() for p in paths]
    

def get_val_reference(val_path='val.txt'):
    path = trainval_reference_dir + '/' + val_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip().split() for p in paths]


def get_test_reference(test_path='test-dev.txt'):
    path = test_reference_dir + '/' + test_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip() for p in paths]


def normalize_image(image):
    return image / 255.0


# min_val, max_val = 35, 255
def data_processing_train():
    
    paths = get_train_reference()
    
    if paths is None:
        raise ValueError("No paths found.")
    
    os.makedirs(f'{output_root_dir}/DAVIS/', exist_ok=True)
    os.makedirs(f'{output_root_dir}/DAVIS/training_slices', exist_ok=True)
    
    for train, label in paths:
        print(train, label)
        img = load_jpg(f'{root_dir}/{train}')
        imgs = normalize_image(img)
        labels = load_jpg(f'{root_dir}/{label}')
        obj_id = train.split('/')[-2]
        slice = train.split('/')[-1].split('.')[0]
        
        assert imgs.shape[0] == labels.shape[0], f"Image and label shapes do not match for {train}"
        assert imgs.shape[1] == labels.shape[1], f"Image and label shapes do not match for {train}"
        
        filename = f'{output_root_dir}/DAVIS/training_slices/{obj_id}_slice_{slice}.h5'
        with h5py.File(filename, 'w') as h5f:
            h5f.create_dataset('image', data=img, dtype='float32')
            h5f.create_dataset('label', data=labels, dtype='uint8')
        
        
def data_processing_val():
    
    paths = get_val_reference()
    
    if paths is None:
        raise ValueError("No paths found.")
    
    os.makedirs(f'{output_root_dir}/DAVIS/', exist_ok=True)
    os.makedirs(f'{output_root_dir}/DAVIS/val_volumes', exist_ok=True)
    
    objects = set()
    for train, _ in paths:
        obj_id = train.split('/')[-2]
        objects.add(obj_id)
    
    img_search_path = glob(f'{root_dir}/JPEGImages/480p/*')
    labels_search_path = glob(f'{root_dir}/Annotations/480p/*')
    
    for obj in objects:
        img_path = [p for p in img_search_path if obj in p]
        label_path = [p for p in labels_search_path if obj in p]

        img_paths = glob(f'{img_path[0]}/*.jpg')
        label_paths = glob(f'{label_path[0]}/*.png')
        
        img_paths = sorted(img_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        label_paths = sorted(label_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        imgs_arr = []
        labels_arr = []
        for img, label in zip(img_paths, label_paths):
            print(img, label)
            imgs = load_jpg(img)
            imgs = normalize_image(imgs)
            labels = load_jpg(label)
        
            assert imgs.shape[0] == labels.shape[0], f"Image and label shapes do not match for {train}"
            assert imgs.shape[1] == labels.shape[1], f"Image and label shapes do not match for {train}"

            # Expand dimension around the first axis
            imgs = np.expand_dims(imgs, axis=0)
            labels = np.expand_dims(labels, axis=0)
            
            imgs_arr.append(imgs)
            labels_arr.append(labels)
        
        imgs_arr = np.vstack(imgs_arr)
        labels_arr = np.vstack(labels_arr)
            
        filename = f'{output_root_dir}/DAVIS/val_volumes/{obj}.h5'
        
        with h5py.File(filename, 'w') as h5f:
            h5f.create_dataset('image', data=imgs_arr, dtype='float32')
            h5f.create_dataset('label', data=labels_arr, dtype='uint8')
            

def data_processing_test():
    
    os.makedirs(f'{output_root_dir}/DAVIS/', exist_ok=True)
    os.makedirs(f'{output_root_dir}/DAVIS/testing_volumes', exist_ok=True)
    
    # img_search_path = glob(f'{root_dir}/DAVIS_test/JPEGImages/480p/*')
    labels_search_path = glob(f'{root_dir}/DAVIS_test/Annotations/480p/*')
    # DATA/DAVIS_2017/DAVIS_test/JPEGImages/480p/aerobatics/00000.jpg
    # DATA/DAVIS_2017/DAVIS_test/Annotations/480p/aerobatics/00000.png
    for i in labels_search_path:
        img_path = i.replace('Annotations', 'JPEGImages').replace('png', 'jpg')

        img_paths = glob(f'{img_path}/*.jpg')
        label_paths = glob(f'{i}/*.png')
        
        img_path = sorted(img_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))[0]
        label_path = sorted(label_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))[0]
        
        imgs = load_jpg(img_path)
        imgs = normalize_image(imgs)
        labels = load_jpg(label_path)
    
        assert imgs.shape[0] == labels.shape[0], f"Image and label shapes do not match for {imgs}"
        assert imgs.shape[1] == labels.shape[1], f"Image and label shapes do not match for {imgs}"

        # Expand dimension around the first axis
        imgs = np.expand_dims(imgs, axis=0)
        labels = np.expand_dims(labels, axis=0)
            
        filename = f'{output_root_dir}/DAVIS/testing_volumes/{i.split("/")[-1]}.h5'
        
        with h5py.File(filename, 'w') as h5f:
            h5f.create_dataset('image', data=imgs, dtype='float32')
            h5f.create_dataset('label', data=labels, dtype='uint8')
            
            
def main():
    data_processing_train()
    data_processing_val()
    data_processing_test()


if __name__ == '__main__':
    main()