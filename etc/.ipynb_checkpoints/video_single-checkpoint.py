import numpy as np
import cv2

img_list = np.load('img_list.npy', allow_pickle=True).tolist()

n = int(len(img_list) / 3)
for i in range(3):
    frames = [cv2.cvtColor(np.array(img[i]), cv2.COLOR_RGB2BGR) for img in img_list]
        
    frame_height, frame_width, _ = frames[0].shape
    fps = 1
    video_filename = f'./video{i}.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        video.write(frame)
    
    video.release()
