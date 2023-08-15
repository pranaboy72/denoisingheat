import numpy as np
import cv2
import imageio
from PIL import Image

img_list = np.load('img_list.npy', allow_pickle=True).tolist()

frames = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in img_list]
    
frame_height, frame_width, _ = frames[0].shape
fps = 50
video_filename = f'./video.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

for frame in frames:
    video.write(frame)

video.release()
# image_list = np.load('./img_list.npy', allow_pickle=True)
# image_array_list = [np.array(img) for img in image_list]
# imageio.mimwrite('./video.mp4', image_array_list, fps=50)
