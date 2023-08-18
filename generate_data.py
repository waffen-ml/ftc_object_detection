import argparse
import os
from PIL import Image
from tqdm import trange
import random
from torchvision import transforms as T
import torch

ROTATE_OBJ = True
MAX_COMMON_RESIZE = 1.7
MIN_COMMON_RESIZE = 0.8
MAX_RESIZE_DIFF = 0.2
MAX_CROP = 0.2
COLOR_JITTER_PROB = 0.8

parser = argparse.ArgumentParser()
parser.add_argument('--obj_folder', default='object_images')
parser.add_argument('--bg_folder', default='bg_images')
parser.add_argument('--output_folder', default='result')
parser.add_argument('--n_images', default=20)
parser.add_argument('--max_displayed_objects', default=4)
args = parser.parse_args()
config = vars(args)

if not os.path.exists(config['output_folder']):
    os.mkdir(config['output_folder'])

obj_images = [Image.open(os.path.join(config['obj_folder'], fname))
              for fname in os.listdir(config['obj_folder'])]

background_fnames = os.listdir(config['bg_folder'])

modules = [
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomPerspective(distortion_scale=0.25, p=0.4),
    T.GaussianBlur(kernel_size=(3, 3))
]

if ROTATE_OBJ:
    modules.append(T.RandomRotation(90))

object_main_transform = torch.nn.Sequential(*modules)
object_color_jitter = T.ColorJitter(brightness=.3, hue=.2, saturation=.2, contrast=.2)

def generate_object():
    image = random.choice(obj_images)
    image = object_main_transform(image)

    c_factor = random.uniform(MIN_COMMON_RESIZE, MAX_COMMON_RESIZE)
    w_factor = c_factor + random.uniform(-MAX_RESIZE_DIFF, MAX_RESIZE_DIFF)
    h_factor = c_factor + random.uniform(-MAX_RESIZE_DIFF, MAX_RESIZE_DIFF)

    image = image.resize((
        int(image.size[0] * w_factor),
        int(image.size[1] * h_factor)
    ))

    crop_type = random.randint(0, 1)
    # 0 -> horizontal
    # 1 -> vertical
    crop_dir = random.randint(0, 1)
    # 0 -> positive
    # 1 -> negative

    max_crop = int(MAX_CROP*(image.size[1] if crop_type else image.size[0]))
    crop = random.randint(0, max_crop)

    xyxy = [0, 0, image.size[0], image.size[1]]

    if not crop_type and not crop_dir:
        # hor pos
        xyxy[2] -= crop
    elif not crop_type and crop_dir:
        # hor neg
        xyxy[0] = crop
    elif crop_type and not crop_dir:
        # ver pos
        xyxy[3] -= crop
    else:
        # ver neg
        xyxy[1] = crop

    image = image.crop(xyxy)

    if random.uniform(0, 1) < COLOR_JITTER_PROB:
        return image, object_color_jitter(image)   

    return image, image


def attach_object(bg, obj, pjitted):
    if obj.size[0] > bg.size[0] or obj.size[1] > bg.size[1]:
        return

    x = random.randint(0, bg.size[0] - obj.size[0])
    y = random.randint(0, bg.size[1] - obj.size[1])

    bg.paste(pjitted, (x, y), obj)

    return (x, y, x + obj.size[0], y + obj.size[1])


def save(i, image, bounds):
    name = 'image' + str(i)
    image.save(os.path.join(config['output_folder'], name + '.jpg'))
    with open(os.path.join(config['output_folder'], name + '.txt'), 'w') as f:
        for b in bounds:
            bstr = ','.join([str(bi) for bi in b]) + ',obj'
            f.write(bstr + '\n')


for i in trange(int(config['n_images'])):
    n_objects = random.randint(0, int(config['max_displayed_objects']))
    bg_fname = random.choice(background_fnames)
    bg = Image.open(os.path.join(config['bg_folder'], bg_fname))
    bounds = []

    for j in range(n_objects):
        transp, pjitted = generate_object()
        b = attach_object(bg, transp, pjitted)
        bounds.append(b)
    
    save(i, bg, bounds)
