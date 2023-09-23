import numpy as np
import cv2
from PIL import Image

img_list = np.load('../results/heat/eval2.npy', allow_pickle=True)

frames = []

for i in range(img_list.shape[0]):
    if isinstance(img_list[i][0], Image.Image):
        img_array = np.array(img_list[i][0])

    else:
        img_array = img_list[i][0][:,:,:3]    

    if img_array.dtype == np.float32:
        img_array = (img_array * 255).astype(np.uint8)
    
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    frames.append(frame)

frame_height, frame_width, _ = frames[0].shape

fps = 100
video_filename = f'./eval.mp4'
    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

for frame in frames:
    video.write(frame)
    
video.release()


# If you want to merge two videos, write following line in the terminal:
# ffmpeg -i eval.mp4 -i heat.mp4 -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" merged.mp4
