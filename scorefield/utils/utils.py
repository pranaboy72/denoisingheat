import os
import glob
import einops
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    
def clear_imgs(path):
    if not os.path.exists(path):
        print("No images in the directory")
    else:
        for file in glob.glob(os.path.join(path, "*.jpg")):
            os.remove(file)
        print("Images deleted successfully.")
        
def random_rgb(goal, num):
    rand_rgbs = []
    while True:
        rand_rgb = np.random.randint(0, 256, size=(3,)).tolist()
        if rand_rgb != goal:
            rand_rgbs = rand_rgbs.append(rand_rgb)
        else:
            continue
        
        if len(rand_rgbs) == num:
            return rand_rgbs
        
def get_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_distractors(goal, bounds, num):
    distractors = []
    
    assert len(bounds) == 4

    while True:
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])
        
        if get_distance([x,y], goal) < 1.:
            continue
        
        if len(distractors) == 0:
            distractors.append([x,y])
        else:
            for i in range(len(distractors)):
                distance = get_distance([x,y], distractors[i])
                if distance < 1.:
                    continue
            distractors.append([x,y])
            
        if len(distractors) == num:
            return distractors
        
def gen_goals(bounds, num):
    goals = []
    fail = False
    assert len(bounds) == 4
    
    while True:    
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])

        if len(goals) == 0:
            goals.append([x,y])
        else:        
            for i in range(len(goals)):
                if get_distance([x,y], goals[i]) < 1.:
                    fail=True
                    break
            if not fail:
                goals.append([x,y])
            fail=False
        
        if len(goals) == num:
            return goals
        
def random_batch(renderer, map_img, batch_size):
    rand_goals = gen_goals(renderer.bounds, batch_size)
    batch = []
    for i in range(batch_size):
        img = renderer.stamp(map_img, rand_goals[i],'goal')
        if img.shape[1] != renderer.img_size:
            img = cv2.resize(img, (renderer.img_size, renderer.img_size), interpolation=cv2.INTER_LANCZOS4)
        if img.shape[-1] == 3:
            img = einops.rearrange(img, 'h w c -> c h w')
        batch.append(img)
    return np.array(batch), rand_goals

def imshow(img):
    plt.imshow(img)
    plt.show()
