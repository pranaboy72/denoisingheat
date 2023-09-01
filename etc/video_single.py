import numpy as np
import cv2

img_list = np.load('../results/heat/eval.npy', allow_pickle=True)

frames = []

for i in range(img_list.shape[0]):
    img_array = img_list[i][0][:,:,:3]    

    if img_array.dtype == np.float32:
        img_array = (img_array * 255).astype(np.uint8)

    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    frames.append(frame)

frame_height, frame_width, _ = frames[0].shape
fps = 20
video_filename = f'./eval.mp4'
    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

for frame in frames:
    video.write(frame)
    
video.release()
