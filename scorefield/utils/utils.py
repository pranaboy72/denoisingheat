import os
import einops
import matplotlib.pyplot as plt
import cv2

def log_num_check(path):
    log_num = 0
    while True:
        new_path = path + '_' + str(log_num) + '.jpg'
        check = os.path.exists(new_path)
        if check:
            log_num += 1
            continue
        else:
            return new_path
    
def visualize(obs):
    obs = einops.rearrange(obs, 'c h w -> h w c')
    plt.imshow(obs)
    plt.show()
    raise

def save_obs(obs):
    if obs.shape[0] == 3:
        obs = einops.rearrange(obs, 'c h w -> h w c')
    if obs.shape[0] != 500:
        cv2.resize(obs, (500,500),interpolation=cv2.INTER_NEAREST)
    img_path = './logs/imgs/img'
    img = log_num_check(img_path)
    
    plt.imsave(img, obs)
    