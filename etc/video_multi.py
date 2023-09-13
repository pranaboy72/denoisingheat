import numpy as np
import cv2
from PIL import Image

img_list = np.load('../results/heat/eval.npy', allow_pickle=True)
heat_list = np.load('../results/heat/heat.npy', allow_pickle=True)
all_list = [img_list, heat_list]

for i, lst in enumerate(all_list):
    frames = []

    for i in range(lst.shape[0]):
        if isinstance(lst[i][0], Image.Image):
            img_array = np.array(lst[i][0])

        else:
            img_array = lst[i][0][:,:,:3]    

        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        frames.append(frame)

    frame_height, frame_width, _ = frames[0].shape

    fps = 30
    video_filename = f'./eval{i}.mp4'
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video.write(frame)
        
    video.release()

