import argparse
import os
import cv2
from PIL import Image
from tqdm import tqdm
 
parser = argparse.ArgumentParser()
parser.add_argument('--gap', default=10)
parser.add_argument('--input_folder', default='bg_videos')
parser.add_argument('--output_folder', default='bg_images')
args = parser.parse_args()
config = vars(args)

frame_gap = int(config['gap'])
input_folder = config['input_folder']
output_folder = config['output_folder']

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files = os.listdir(input_folder)

for file in tqdm(files):
    try:
        i = file.rindex('.')
    except:
        continue
    name = file[:i]
    format = file[i + 1:]
    if format != 'mp4':
        continue
    cap = cv2.VideoCapture(os.path.join(input_folder, file))
    if not cap.isOpened():
        continue
    n = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if (n - 1) % frame_gap:
            continue
        frame = frame[:, :, [2, 1, 0]]
        t = Image.fromarray(frame)
        t.save(os.path.join(output_folder, f'{name}_frame{n - 1}.jpg'))


