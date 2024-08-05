import os 
from glob import glob 
from PIL import Image
import numpy as np
import h5py


root_dir = '/Users/admin/Downloads/DAVIS_2017'
trainval_reference_dir = '/Users/admin/Downloads/DAVIS_2017/ImageSets/2016'
image_dir = '/Users/admin/Downloads/DAVIS_2017/JPEGImages/480p'
ann_dir = '/Users/admin/Downloads/DAVIS_2017/Annotations/480p'
test_reference_dir = './DATA/DAVIS_2017/DAVIS_test/ImageSets/2017'
output_root_dir = './OUTPUTS'


def load_jpg(image_path):
    # Load the image
    image = Image.open(image_path)
    return np.array(image)

def get_obj_super_cat_id_map():
    super_cat_id_map = get_super_cat_id_map()
    obj_map = {obj_name: super_cat_id_map[obj_map['super_category']] for obj_name, obj_map in categories_map.items()}
    return obj_map

def get_super_cat_id_map():
    super_cats = get_super_cat_set()
    super_cats = sorted(list(super_cats))
    # make sure to start at for label map because 0 is reserved for background
    label_map = {cat: (int_id+1) for int_id, cat in enumerate(super_cats)}
    return label_map


def get_super_cat_set():
    super_categories = set()
    for k, v in categories_map.items():
        super_categories.add(v['super_category'])
    return super_categories


# DATA/DAVIS_2017/JPEGImages/480p/lucia/00011.jpg
# DATA/DAVIS_2017/Annotations/480p/lucia/00011.png
def get_train_reference(train_path='train.txt'):
    path = trainval_reference_dir + '/' + train_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip() for p in paths]
    

def get_val_reference(val_path='val.txt'):
    path = trainval_reference_dir + '/' + val_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip() for p in paths]


def get_test_reference(test_path='test-dev.txt'):
    path = test_reference_dir + '/' + test_path
    with open(path, 'r') as file:
        paths = file.readlines()
    return [p.strip() for p in paths]


def normalize_image(image):
    return image / 255.0


# min_val, max_val = 35, 255
def data_processing_train(obj_super_cat_id_map):
    
    paths = get_train_reference()
    
    if paths is None:
        raise ValueError("No paths found.")
    
    os.makedirs(f'{output_root_dir}/DAVIS/', exist_ok=True)
    os.makedirs(f'{output_root_dir}/DAVIS/training_slices', exist_ok=True)
    for obj in paths:
        image_frames = sorted(glob(os.path.join(image_dir, obj, "*")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        ann_frames = sorted(glob(os.path.join(ann_dir, obj, "*")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for image_frame, ann_frame in zip(image_frames, ann_frames):
            print(image_frame, ann_frame)
            img = load_jpg(image_frame)
            imgs = normalize_image(img)
            labels = load_jpg(ann_frame)
            print("before")
            print(np.unique(labels))
            video_name = image_frame.split('/')[-2]
            slice = image_frame.split('/')[-1].split('.')[0]
            print("video_name: {}".format(video_name))
            print("slice: {}".format(slice))
            obj_map = davis_semantics_map[video_name]
            for obj_id, obj_name in obj_map.items():
                labels[labels == int(obj_id)] = obj_super_cat_id_map[obj_name]
            print("after")
            print(np.unique(labels))

            assert imgs.shape[0] == labels.shape[0], f"Image and label shapes do not match for {image_frame}"
            assert imgs.shape[1] == labels.shape[1], f"Image and label shapes do not match for {image_frame}"

            filename = f'{output_root_dir}/DAVIS/training_slices/{obj_id}_slice_{slice}.h5'
            with h5py.File(filename, 'w') as h5f:
                h5f.create_dataset('image', data=img, dtype='float32')
                h5f.create_dataset('label', data=labels, dtype='uint8')


def data_processing_val(obj_super_cat_id_map):
    
    paths = get_val_reference()
    
    if paths is None:
        raise ValueError("No paths found.")
    
    os.makedirs(f'{output_root_dir}/DAVIS/', exist_ok=True)
    os.makedirs(f'{output_root_dir}/DAVIS/val_volumes', exist_ok=True)

    for obj in paths:
        print(obj)
        imgs_arr = []
        labels_arr = []
        image_frames = sorted(glob(os.path.join(image_dir, obj, "*")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        ann_frames = sorted(glob(os.path.join(ann_dir, obj, "*")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for image_frame, ann_frame in zip(image_frames, ann_frames):
            img = load_jpg(image_frame)
            imgs = normalize_image(img)
            labels = load_jpg(ann_frame)
            video_name = image_frame.split('/')[-2]
            slice = image_frame.split('/')[-1].split('.')[0]
            obj_map = davis_semantics_map[video_name]
            for obj_id, obj_name in obj_map.items():
                labels[labels == int(obj_id)] = obj_super_cat_id_map[obj_name]

            assert imgs.shape[0] == labels.shape[0], f"Image and label shapes do not match for {image_frame}"
            assert imgs.shape[1] == labels.shape[1], f"Image and label shapes do not match for {image_frame}"

            imgs_arr.append(imgs)
            labels_arr.append(labels)
        
        imgs_arr = np.stack(imgs_arr, axis=0)
        labels_arr = np.stack(labels_arr, axis=0)

        print(imgs_arr.shape)

        filename = f'{output_root_dir}/DAVIS/val_volumes/{obj}.h5'

        with h5py.File(filename, 'w') as h5f:
            h5f.create_dataset('image', data=imgs_arr, dtype='float32')
            h5f.create_dataset('label', data=labels_arr, dtype='uint8')


categories_map = {
    "airplane": {
        "id": 1,
        "super_category": "vehicle"
    },
    "backpack": {
        "id": 2,
        "super_category": "accessory"
    },
    "ball": {
        "id": 3,
        "super_category": "sports"
    },
    "bear": {
        "id": 4,
        "super_category": "animal"
    },
    "bicycle": {
        "id": 5,
        "super_category": "vehicle"
    },
    "bird": {
        "id": 6,
        "super_category": "animal"
    },
    "boat": {
        "id": 7,
        "super_category": "vehicle"
    },
    "bottle": {
        "id": 8,
        "super_category": "kitchen"
    },
    "box": {
        "id": 9,
        "super_category": "device"
    },
    "bus": {
        "id": 10,
        "super_category": "vehicle"
    },
    "camel": {
        "id": 11,
        "super_category": "animal"
    },
    "car": {
        "id": 12,
        "super_category": "vehicle"
    },
    "carriage": {
        "id": 13,
        "super_category": "vehicle"
    },
    "cat": {
        "id": 14,
        "super_category": "animal"
    },
    "cellphone": {
        "id": 15,
        "super_category": "electronic"
    },
    "chamaleon": {
        "id": 16,
        "super_category": "animal"
    },
    "cow": {
        "id": 17,
        "super_category": "animal"
    },
    "deer": {
        "id": 18,
        "super_category": "animal"
    },
    "dog": {
        "id": 19,
        "super_category": "animal"
    },
    "dolphin": {
        "id": 20,
        "super_category": "animal"
    },
    "drone": {
        "id": 21,
        "super_category": "electronic"
    },
    "elephant": {
        "id": 22,
        "super_category": "animal"
    },
    "excavator": {
        "id": 23,
        "super_category": "vehicle"
    },
    "fish": {
        "id": 24,
        "super_category": "animal"
    },
    "goat": {
        "id": 25,
        "super_category": "animal"
    },
    "golf cart": {
        "id": 26,
        "super_category": "vehicle"
    },
    "golf club": {
        "id": 27,
        "super_category": "sports"
    },
    "grass": {
        "id": 28,
        "super_category": "outdoor"
    },
    "guitar": {
        "id": 29,
        "super_category": "instrument"
    },
    "gun": {
        "id": 30,
        "super_category": "sports"
    },
    "helicopter": {
        "id": 31,
        "super_category": "vehicle"
    },
    "horse": {
        "id": 32,
        "super_category": "animal"
    },
    "hoverboard": {
        "id": 33,
        "super_category": "sports"
    },
    "kart": {
        "id": 34,
        "super_category": "vehicle"
    },
    "key": {
        "id": 35,
        "super_category": "device"
    },
    "kite": {
        "id": 36,
        "super_category": "sports"
    },
    "koala": {
        "id": 37,
        "super_category": "animal"
    },
    "leash": {
        "id": 38,
        "super_category": "device"
    },
    "lion": {
        "id": 39,
        "super_category": "animal"
    },
    "lock": {
        "id": 40,
        "super_category": "device"
    },
    "mask": {
        "id": 41,
        "super_category": "accessory"
    },
    "microphone": {
        "id": 42,
        "super_category": "electronic"
    },
    "monkey": {
        "id": 43,
        "super_category": "animal"
    },
    "motorcycle": {
        "id": 44,
        "super_category": "vehicle"
    },
    "oar": {
        "id": 45,
        "super_category": "sports"
    },
    "paper": {
        "id": 46,
        "super_category": "device"
    },
    "paraglide": {
        "id": 47,
        "super_category": "sports"
    },
    "person": {
        "id": 48,
        "super_category": "person"
    },
    "pig": {
        "id": 49,
        "super_category": "animal"
    },
    "pole": {
        "id": 50,
        "super_category": "sports"
    },
    "potted plant": {
        "id": 51,
        "super_category": "furniture"
    },
    "puck": {
        "id": 52,
        "super_category": "sports"
    },
    "rack": {
        "id": 53,
        "super_category": "furniture"
    },
    "rhino": {
        "id": 54,
        "super_category": "animal"
    },
    "rope": {
        "id": 55,
        "super_category": "sports"
    },
    "sail": {
        "id": 56,
        "super_category": "sports"
    },
    "scale": {
        "id": 57,
        "super_category": "appliance"
    },
    "scooter": {
        "id": 58,
        "super_category": "vehicle"
    },
    "selfie stick": {
        "id": 59,
        "super_category": "device"
    },
    "sheep": {
        "id": 60,
        "super_category": "animal"
    },
    "skateboard": {
        "id": 61,
        "super_category": "sports"
    },
    "ski": {
        "id": 62,
        "super_category": "sports"
    },
    "ski poles": {
        "id": 63,
        "super_category": "sports"
    },
    "snake": {
        "id": 64,
        "super_category": "animal"
    },
    "snowboard": {
        "id": 65,
        "super_category": "sports"
    },
    "stick": {
        "id": 66,
        "super_category": "sports"
    },
    "stroller": {
        "id": 67,
        "super_category": "vehicle"
    },
    "surfboard": {
        "id": 68,
        "super_category": "sports"
    },
    "swing": {
        "id": 69,
        "super_category": "outdoor"
    },
    "tennis racket": {
        "id": 70,
        "super_category": "sports"
    },
    "tractor": {
        "id": 71,
        "super_category": "vehicle"
    },
    "trailer": {
        "id": 72,
        "super_category": "vehicle"
    },
    "train": {
        "id": 73,
        "super_category": "vehicle"
    },
    "truck": {
        "id": 74,
        "super_category": "vehicle"
    },
    "turtle": {
        "id": 75,
        "super_category": "animal"
    },
    "varanus": {
        "id": 76,
        "super_category": "animal"
    },
    "violin": {
        "id": 77,
        "super_category": "instrument"
    },
    "wheelchair": {
        "id": 78,
        "super_category": "vehicle"
    }
}

davis_semantics_map = {
    "aerobatics": {
        "1": "airplane",
        "2": "selfie stick",
        "3": "person"
    },
    "bear": {
        "1": "bear"
    },
    "bike-packing": {
        "1": "bicycle",
        "2": "person"
    },
    "bike-trial": {
        "1": "person",
        "2": "bicycle"
    },
    "blackswan": {
        "1": "bird"
    },
    "bmx-bumps": {
        "1": "bicycle",
        "2": "person"
    },
    "bmx-trees": {
        "1": "bicycle",
        "2": "person"
    },
    "boat": {
        "1": "boat"
    },
    "boxing": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "boxing-fisheye": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "breakdance": {
        "1": "person"
    },
    "breakdance-flare": {
        "1": "person"
    },
    "burnout": {
        "1": "car"
    },
    "bus": {
        "1": "bus"
    },
    "camel": {
        "1": "camel"
    },
    "car-race": {
        "1": "person",
        "2": "car",
        "3": "car",
        "4": "car"
    },
    "car-roundabout": {
        "1": "car"
    },
    "car-shadow": {
        "1": "car"
    },
    "car-turn": {
        "1": "car"
    },
    "carousel": {
        "1": "horse",
        "2": "horse",
        "3": "horse",
        "4": "horse"
    },
    "cat-girl": {
        "1": "person",
        "2": "cat"
    },
    "cats-car": {
        "1": "cat",
        "2": "cat",
        "3": "person",
        "4": "person"
    },
    "chamaleon": {
        "1": "chamaleon"
    },
    "choreography": {
        "1": "person",
        "2": "person",
        "3": "person",
        "4": "person",
        "5": "person",
        "6": "person",
        "7": "person"
    },
    "classic-car": {
        "1": "person",
        "2": "person",
        "3": "car"
    },
    "color-run": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "cows": {
        "1": "cow"
    },
    "crossing": {
        "1": "person",
        "2": "person",
        "3": "truck"
    },
    "dance-jump": {
        "1": "person"
    },
    "dance-twirl": {
        "1": "person"
    },
    "dancing": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "deer": {
        "1": "deer",
        "2": "deer"
    },
    "demolition": {
        "1": "excavator",
        "2": "excavator",
        "3": "excavator"
    },
    "disc-jockey": {
        "1": "person",
        "2": "person",
        "3": "mask"
    },
    "dive-in": {
        "1": "person"
    },
    "dog": {
        "1": "dog"
    },
    "dog-agility": {
        "1": "dog"
    },
    "dog-control": {
        "1": "leash",
        "2": "dog",
        "3": "microphone",
        "4": "person",
        "5": "person"
    },
    "dog-gooses": {
        "1": "dog",
        "2": "bird",
        "3": "bird",
        "4": "bird",
        "5": "bird"
    },
    "dogs-jump": {
        "1": "dog",
        "2": "dog",
        "3": "person"
    },
    "dogs-scale": {
        "1": "dog",
        "2": "dog",
        "3": "person",
        "4": "scale"
    },
    "dolphins": {
        "1": "dolphin",
        "2": "dolphin",
        "3": "dolphin",
        "4": "dolphin",
        "5": "dolphin",
        "6": "dolphin"
    },
    "drift-chicane": {
        "1": "car"
    },
    "drift-straight": {
        "1": "car"
    },
    "drift-turn": {
        "1": "car"
    },
    "drone": {
        "1": "drone",
        "2": "cellphone",
        "3": "person",
        "4": "cellphone",
        "5": "person"
    },
    "e-bike": {
        "1": "person",
        "2": "bicycle"
    },
    "elephant": {
        "1": "elephant"
    },
    "flamingo": {
        "1": "bird"
    },
    "giant-slalom": {
        "1": "ski",
        "2": "ski poles",
        "3": "person"
    },
    "girl-dog": {
        "1": "person",
        "2": "dog",
        "3": "wheelchair"
    },
    "goat": {
        "1": "goat"
    },
    "gold-fish": {
        "1": "fish",
        "2": "fish",
        "3": "fish",
        "4": "fish",
        "5": "fish"
    },
    "golf": {
        "1": "golf cart",
        "2": "golf club",
        "3": "person"
    },
    "grass-chopper": {
        "1": "tractor",
        "2": "trailer",
        "3": "grass",
        "4": "tractor"
    },
    "guitar-violin": {
        "1": "guitar",
        "2": "person",
        "3": "person",
        "4": "violin"
    },
    "gym": {
        "1": "person",
        "2": "person",
        "3": "bottle",
        "4": "bottle"
    },
    "helicopter": {
        "1": "person",
        "2": "helicopter"
    },
    "hike": {
        "1": "person"
    },
    "hockey": {
        "1": "person",
        "2": "stick",
        "3": "puck"
    },
    "horsejump-high": {
        "1": "horse",
        "2": "person"
    },
    "horsejump-low": {
        "1": "horse",
        "2": "person"
    },
    "horsejump-stick": {
        "1": "stick",
        "2": "person",
        "3": "horse"
    },
    "hoverboard": {
        "1": "person",
        "2": "hoverboard"
    },
    "hurdles": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "india": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "inflatable": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "judo": {
        "1": "person",
        "2": "person"
    },
    "juggle": {
        "1": "ball",
        "2": "person",
        "3": "guitar",
        "4": "person"
    },
    "kart-turn": {
        "1": "person",
        "2": "kart",
        "3": "person",
        "4": "kart"
    },
    "kid-football": {
        "1": "person",
        "2": "ball"
    },
    "kids-turning": {
        "1": "person",
        "2": "person"
    },
    "kite-surf": {
        "1": "sail",
        "2": "surfboard",
        "3": "person"
    },
    "kite-walk": {
        "1": "kite",
        "2": "person",
        "3": "surfboard"
    },
    "koala": {
        "1": "koala"
    },
    "lab-coat": {
        "1": "cellphone",
        "2": "cellphone",
        "3": "person",
        "4": "person",
        "5": "person"
    },
    "lady-running": {
        "1": "person",
        "2": "paper"
    },
    "libby": {
        "1": "dog"
    },
    "lindy-hop": {
        "1": "person",
        "2": "person",
        "3": "person",
        "4": "person",
        "5": "person",
        "6": "person",
        "7": "person",
        "8": "person"
    },
    "lions": {
        "1": "lion",
        "2": "lion",
        "3": "lion"
    },
    "loading": {
        "1": "person",
        "2": "box",
        "3": "person"
    },
    "lock": {
        "1": "person",
        "2": "person",
        "3": "key",
        "4": "lock"
    },
    "longboard": {
        "1": "backpack",
        "2": "bicycle",
        "3": "skateboard",
        "4": "person",
        "5": "person"
    },
    "lucia": {
        "1": "person"
    },
    "mallard-fly": {
        "1": "bird"
    },
    "mallard-water": {
        "1": "bird"
    },
    "man-bike": {
        "1": "person",
        "2": "motorcycle"
    },
    "mbike-santa": {
        "1": "person",
        "2": "motorcycle"
    },
    "mbike-trick": {
        "1": "person",
        "2": "motorcycle"
    },
    "miami-surf": {
        "1": "person",
        "2": "surfboard",
        "3": "person",
        "4": "surfboard",
        "5": "person",
        "6": "surfboard"
    },
    "monkeys": {
        "1": "monkey",
        "2": "monkey"
    },
    "monkeys-trees": {
        "1": "monkey",
        "2": "monkey"
    },
    "motocross-bumps": {
        "1": "person",
        "2": "motorcycle"
    },
    "motocross-jump": {
        "1": "person",
        "2": "motorcycle"
    },
    "motorbike": {
        "1": "motorcycle",
        "2": "person",
        "3": "person"
    },
    "mtb-race": {
        "1": "person",
        "2": "bicycle",
        "3": "bicycle",
        "4": "person"
    },
    "night-race": {
        "1": "car",
        "2": "car"
    },
    "ocean-birds": {
        "1": "bird",
        "2": "bird"
    },
    "orchid": {
        "1": "potted plant",
        "2": "person"
    },
    "paragliding": {
        "1": "person",
        "2": "paraglide"
    },
    "paragliding-launch": {
        "1": "backpack",
        "2": "person",
        "3": "paraglide"
    },
    "parkour": {
        "1": "person"
    },
    "people-sunset": {
        "1": "person",
        "2": "person",
        "3": "person",
        "4": "person"
    },
    "pigs": {
        "1": "pig",
        "2": "pig",
        "3": "pig"
    },
    "planes-crossing": {
        "1": "airplane",
        "2": "airplane"
    },
    "planes-water": {
        "1": "airplane",
        "2": "airplane"
    },
    "pole-vault": {
        "1": "pole",
        "2": "person"
    },
    "rallye": {
        "1": "car"
    },
    "rhino": {
        "1": "rhino"
    },
    "rollerblade": {
        "1": "person"
    },
    "rollercoaster": {
        "1": "kart"
    },
    "running": {
        "1": "person",
        "2": "person"
    },
    "salsa": {
        "1": "person",
        "10": "person",
        "2": "person",
        "3": "person",
        "4": "person",
        "5": "person",
        "6": "person",
        "7": "person",
        "8": "person",
        "9": "person"
    },
    "schoolgirls": {
        "1": "person",
        "2": "backpack",
        "3": "person",
        "4": "backpack",
        "5": "person",
        "6": "backpack",
        "7": "person"
    },
    "scooter-black": {
        "1": "person",
        "2": "motorcycle"
    },
    "scooter-board": {
        "1": "person",
        "2": "scooter"
    },
    "scooter-gray": {
        "1": "motorcycle",
        "2": "person"
    },
    "seasnake": {
        "1": "snake"
    },
    "selfie": {
        "1": "person",
        "2": "person",
        "3": "person",
        "4": "cellphone",
        "5": "person"
    },
    "sheep": {
        "1": "sheep",
        "2": "sheep",
        "3": "sheep",
        "4": "sheep",
        "5": "sheep"
    },
    "shooting": {
        "1": "gun",
        "2": "person",
        "3": "rope"
    },
    "skate-jump": {
        "1": "skateboard",
        "2": "person"
    },
    "skate-park": {
        "1": "person",
        "2": "skateboard"
    },
    "skydive": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "slackline": {
        "1": "person"
    },
    "snowboard": {
        "1": "snowboard",
        "2": "person"
    },
    "soapbox": {
        "1": "kart",
        "2": "person",
        "3": "person"
    },
    "soccerball": {
        "1": "ball"
    },
    "speed-skating": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "stroller": {
        "1": "person",
        "2": "stroller"
    },
    "stunt": {
        "1": "scooter",
        "2": "person"
    },
    "subway": {
        "1": "person",
        "2": "person",
        "3": "person",
        "4": "person"
    },
    "surf": {
        "1": "surfboard",
        "2": "person",
        "3": "sail"
    },
    "swing": {
        "1": "person",
        "2": "swing",
        "3": "leash"
    },
    "swing-boy": {
        "1": "swing",
        "2": "person"
    },
    "tackle": {
        "1": "ball",
        "2": "person",
        "3": "person"
    },
    "tandem": {
        "1": "selfie stick",
        "2": "paraglide",
        "3": "person",
        "4": "person"
    },
    "tennis": {
        "1": "person",
        "2": "tennis racket"
    },
    "tennis-vest": {
        "1": "tennis racket",
        "2": "person"
    },
    "tractor": {
        "1": "tractor",
        "2": "trailer"
    },
    "tractor-sand": {
        "1": "trailer",
        "2": "person",
        "3": "tractor"
    },
    "train": {
        "1": "carriage",
        "2": "carriage",
        "3": "carriage",
        "4": "train"
    },
    "tuk-tuk": {
        "1": "person",
        "2": "person",
        "3": "person"
    },
    "turtle": {
        "1": "turtle"
    },
    "upside-down": {
        "1": "person",
        "2": "person"
    },
    "varanus-cage": {
        "1": "varanus"
    },
    "varanus-tree": {
        "1": "leash",
        "2": "varanus"
    },
    "vietnam": {
        "1": "person",
        "2": "oar",
        "3": "person",
        "4": "oar",
        "5": "person",
        "6": "oar",
        "7": "person"
    },
    "walking": {
        "1": "person",
        "2": "person"
    },
    "wings-turn": {
        "1": "rack",
        "2": "person",
        "3": "person"
    }
}
            
            
def main():
    super_cat_id_map = get_super_cat_id_map()
    print(super_cat_id_map)
    obj_super_cat_id_map = get_obj_super_cat_id_map()
    print(obj_super_cat_id_map)
    #data_processing_train(obj_super_cat_id_map)
    data_processing_val(obj_super_cat_id_map)

"""
{'accessory': 1, 'animal': 2, 'appliance': 3, 'device': 4, 'electronic': 5, 'furniture': 6, 'instrument': 7, 'kitchen': 8, 'outdoor': 9, 'person': 10, 'sports': 11, 'vehicle': 12}
{'airplane': 12, 'backpack': 1, 'ball': 11, 'bear': 2, 'bicycle': 12, 'bird': 2, 'boat': 12, 'bottle': 8, 'box': 4, 'bus': 12, 'camel': 2, 'car': 12, 'carriage': 12, 'cat': 2, 'cellphone': 5, 'chamaleon': 2, 'cow': 2, 'deer': 2, 'dog': 2, 'dolphin': 2, 'drone': 5, 'elephant': 2, 'excavator': 12, 'fish': 2, 'goat': 2, 'golf cart': 12, 'golf club': 11, 'grass': 9, 'guitar': 7, 'gun': 11, 'helicopter': 12, 'horse': 2, 'hoverboard': 11, 'kart': 12, 'key': 4, 'kite': 11, 'koala': 2, 'leash': 4, 'lion': 2, 'lock': 4, 'mask': 1, 'microphone': 5, 'monkey': 2, 'motorcycle': 12, 'oar': 11, 'paper': 4, 'paraglide': 11, 'person': 10, 'pig': 2, 'pole': 11, 'potted plant': 6, 'puck': 11, 'rack': 6, 'rhino': 2, 'rope': 11, 'sail': 11, 'scale': 3, 'scooter': 12, 'selfie stick': 4, 'sheep': 2, 'skateboard': 11, 'ski': 11, 'ski poles': 11, 'snake': 2, 'snowboard': 11, 'stick': 11, 'stroller': 12, 'surfboard': 11, 'swing': 9, 'tennis racket': 11, 'tractor': 12, 'trailer': 12, 'train': 12, 'truck': 12, 'turtle': 2, 'varanus': 2, 'violin': 7, 'wheelchair': 12}
"""


if __name__ == '__main__':
    main()