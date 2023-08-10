import os
import glob
import einops
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL
from PIL import Image, ImageDraw
import requests
from typing import Optional, Union
import random


def log_num_check(path):
    log_num = 0
    file_type = path.split('.')[-1]
    while True:
        new_path = path[:-3] + '_' + str(log_num) + '.' + file_type
        check = os.path.exists(new_path)
        if check:
            log_num += 1
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
    return torch.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

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
        
        
def gen_goals(bounds, n:Union[tuple, int], dist:Optional[float]=None, device='cuda'):
    assert len(bounds) == 4, f'Unappropriate map bound: {bounds}'
    if isinstance(n, int):
        M = 1
        N = n
    else:
        M, N = n
        
    goals = []
    for m in range(M):
        while True:
            x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand((N, 1), dtype=torch.float32, device=device)
            y = bounds[2] + (bounds[3] - bounds[2]) * torch.rand((N, 1), dtype=torch.float32, device=device)

            if dist is None:
                goals.append(torch.cat((x, y), dim=1))
                break
            else:
                valid=True
                for i in range(N):
                    for j in range(i+1, N):
                        if get_distance([x[i],y[i]], [x[j], y[j]]) < dist:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    goals.append(torch.cat((x, y), dim=1))
                    break
                    
    if M == 1:
        return torch.stack(goals).squeeze(0)
    return torch.stack(goals)

def make_batch(renderer, map_img, targets, batch_size):
    batch=[]
    for i in range(batch_size):
        img = renderer.stamp(map_img, targets[i],'goal')
        if img.shape[1] != renderer.img_size:
            img = cv2.resize(img, (renderer.img_size, renderer.img_size), interpolation=cv2.INTER_LANCZOS4)
        if img.shape[-1] == 3:
            img = einops.rearrange(img, 'h w c -> c h w')
        batch.append(img)
    return np.array(batch)
        
def random_batch(renderer, map_img, batch_size):
    rand_goals = gen_goals(renderer.bounds, batch_size)
    return make_batch(renderer, map_img, rand_goals, batch_size), rand_goals

def eval_batch(renderer, map_img, target, batch_size):
    return make_batch(renderer, map_img, target, batch_size)

def imshow(img):
    plt.imshow(img)
    plt.show()

def prepare_input(img, img_size, goal_pos, circle_rad: float=5):
    assert goal_pos.ndim ==2 and goal_pos.shape[-1] ==2, f"{goal_pos.shape}" # (N,2)
    
    if img.height != img_size:
        new_size = (img_size, img_size)
        img = img.resize(new_size, Image.BICUBIC)
    
    W,H = img.width, img.height
    
    goal_pos_pix = ((1+goal_pos)/2 * torch.tensor([H,W], device=goal_pos.device, dtype=goal_pos.dtype))
    
    imgs = []
    for center in goal_pos_pix.cpu().numpy():
        img_new = img.copy()
        w,h = center[1], center[0]
        
        draw = ImageDraw.Draw(img_new)
        draw.ellipse((w-circle_rad, h-circle_rad, w+circle_rad, h+circle_rad), fill = 'red', outline='red')
        
        img_np = np.array(img_new)[...,:3] / 255
        imgs.append(img_np)
        
    imgs = np.stack(imgs, axis=0)
    return torch.tensor(imgs, dtype=goal_pos.dtype, device=goal_pos.device).permute(0, 3, 1, 2)

 
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.Lambda.ToPILImage(),
    ])
    
    if len(image.shape) == 4:
        image = image[0,:,:,:]
    plt.imshow(reverse_transforms(image))
    
@torch.no_grad()
def sample_plot_image(model, diffusion, map_img, device='cuda'):
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    T = diffusion.noise_steps
    stepsize = int(T/num_images)
    img_size = np.array(map_img).shape[1]
   
    x_trace = []
    x = torch.tensor([-0.5, 0.5]*20, device=device, dtype=torch.float32)

    for i in range(0, T):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        obs = prepare_input(img_size, map_img, goal_pos=x)
        x = diffusion.sample_onestep(model, obs, x, t)
        
        obs = prepare_input(img_size, map_img, goal_pos=x)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(obs.detach().cpu())
            
    plt.show()

def draw_samples(img, goal_pos, circle_rad:float=2):
    W, H = img.width, img.height
    
    goal_pos_pix = ((1+goal_pos)/2 * torch.tensor([H,W], device=goal_pos.device, dtype=goal_pos.dtype))
    img_new = img.copy()
    draw = ImageDraw.Draw(img_new)
    for center in goal_pos_pix.cpu().numpy():
        w,h = center[1],center[0]
        draw.ellipse((w-circle_rad, h-circle_rad, w+circle_rad, h+circle_rad), fill = 'red', outline='red')
    return img_new

def draw_goal_samples(img, goal_pos, current_pos, circle_rad:float=2):
    W, H = img.width, img.height
    
    goal_pos_pix = ((1+goal_pos)/2 * torch.tensor([H,W], device=goal_pos.device, dtype=goal_pos.dtype))
    img_new = img.copy()
    draw = ImageDraw.Draw(img_new)
    for center in goal_pos_pix.cpu().numpy():
        w,h = center[1],center[0]
        draw.ellipse((w-circle_rad, h-circle_rad, w+circle_rad, h+circle_rad), fill = 'red', outline='red')
    
    current_pos_pix = ((1+current_pos)/2 * torch.tensor([H,W], device=current_pos.device, dtype=current_pos.dtype))
    for center in current_pos_pix.cpu().numpy():
        w,h = center[1],center[0]
        draw.ellipse((w-circle_rad, h-circle_rad, w+circle_rad, h+circle_rad), fill = 'blue', outline='blue')
    
    return img_new


# garbage: 'https://e7.pngegg.com/pngimages/459/226/png-clipart-brown-cardboard-boxes-with-black-trash-bags-and-garbage-waste-collection-household-hazardous-waste-house-clearance-waste-management-others-miscellaneous-recycling.png'
def get_url_image(url, name):
    png = requests.get(url)

    with open(f'{name}.png', 'wb') as f:
        f.write(png.content)
        
def get_url_pretrained(url, pt):
    response = requests.get(url, stream=True)
    assert response.status_code == 200  # check if the request was successful

#     with open(pt, 'wb') as f:
#         for chunk in response.iter_content(chunk_size=1024):
#             f.write(chunk)
    with open(pt, 'rb') as f:
        print(f.read(1000))  # Print the first 1000 bytes of the file

    model_state_dict = torch.load(pt)
    return model_state_dict
        
def overlay_image(img, img_size, objs, pos):
    if img.height != img_size:
        new_size = (img_size, img_size)
        img = img.resize(new_size, Image.LANCZOS)
    
    W, H = img.size
    
    for i in range(len(objs)):
        objs[i] = objs[i].resize((W // 5, H // 5), Image.LANCZOS)
    
    pos_pix = ((1+pos)/2 * torch.tensor([H, W], device=pos.device, dtype=pos.dtype)).squeeze(0)
    
    imgs = []
    for i, center in enumerate(pos_pix.cpu().numpy()):
        bg = img.copy()
        obj_num = i % len(objs)
        c0, c1 = round(center[1]),round(center[0])
        w, h = objs[obj_num].size
        bg.paste(objs[obj_num], (c0 - w//2, c1 - h//2), objs[obj_num])
        img_np = np.array(bg)[...,:3] / 255
        imgs.append(img_np)
        
    imgs = np.stack(imgs, axis=0)
    
    return torch.tensor(imgs, dtype=pos.dtype, device=pos.device).permute(0, 3, 1, 2)


def overlay_images(img, img_size, objs, pos, n:Optional[list]=None):
    obs = objs.copy()
    if img.height != img_size:
        new_size = (img_size, img_size)
        img = img.resize(new_size, Image.ANTIALIAS)
    
    while len(obs) != n and n is not None:
        idx = random.randrange(len(obs))
        obs.pop(idx)
    
    W, H = img.size
    
    for i in range(len(obs)):
        obs[i] = obs[i].resize((W // 5, H // 5), Image.LANCZOS)
        
    pos_pix = ((1+pos)/2 * torch.tensor([H, W], device=pos.device, dtype=pos.dtype))
    
    imgs = []
    for i, center in enumerate(pos_pix.cpu().numpy()):
        obj_num = i % len(obs)
        if obj_num == 0: bg = img.copy()
        c0, c1 = round(center[1]), round(center[0])
        w, h = obs[obj_num].size
        bg.paste(obs[obj_num], (c0 - w//2, c1 - h//2), obs[obj_num])
        
        if obj_num == len(obs)-1:
            img_np = np.array(bg)[...,:3] / 255
            imgs.append(img_np)
    imgs = np.stack(imgs, axis=0)
    
    return torch.tensor(imgs, dtype=pos.dtype, device=pos.device).permute(0, 3, 1, 2)
    
    
def overlay_goal_agent(bg, obj, goal, agent, circle_rad:float=3):
    bg = bg.copy()
    W, H = bg.size
    
    obj = [ob.resize((W // 5, H // 5), Image.LANCZOS) for ob in obj]
    
    goal_pix = ((1+goal)/2 * torch.tensor([H, W], device=goal.device, dtype=goal.dtype))
    agent_pix = ((1 + agent) / 2 * torch.tensor([H, W], device=agent.device, dtype=agent.dtype).reshape(1, 1, 2))
#     agent_pix = ((1+agent)/2 * torch.tensor([H, W], device=agent.device, dtype=agent.dtype))

    assert goal_pix.dim() == 2  # (B, 2)
    assert agent_pix.dim() == 3  # (B, N, 2)

    bgs = []
    for i, center in enumerate(goal_pix.cpu().numpy()):
        c0, c1 = round(center[1]),round(center[0])
        w, h = obj[i].size 
        bg_copy = bg.copy()
        bg_copy.paste(obj[i], (c0 - w//2, c1 - h//2), obj[i])
        bgs.append(bg_copy)
        
    draws = [ImageDraw.Draw(bg) for bg in bgs]
    for i, center in enumerate(agent_pix.cpu().numpy()):
        for cen in center:
            c0, c1 = round(cen[1]), round(cen[0])
            draws[i].ellipse((c0-circle_rad, c1-circle_rad, c0+circle_rad, c1+circle_rad), fill = 'red', outline='red')
        
    return bgs
    
def overlay_goals_agent(bg, obj, goal, agent, circle_rad:float=5):
    bg = bg.copy()
    W, H = bg.size
    
    obj = [ob.resize((W // 5, H // 5), Image.LANCZOS) for ob in obj]
    
    goals_pix = ((1+goal)/2 * torch.tensor([[H, W]]*goal.size(0), device=goal.device, dtype=goal.dtype))
    agent_pix = ((1+agent)/2 * torch.tensor([H, W], device=agent.device, dtype=agent.dtype))

    if agent_pix.dim() == 1:
        agent_pix = agent_pix.unsqueeze(0)

    for i, center in enumerate(goals_pix.cpu().numpy()):
        c0, c1 = round(center[1]),round(center[0])
        w, h = obj[i].size
        bg.paste(obj[i], (c0 - w//2, c1 - h//2), obj[i])
        
    draw = ImageDraw.Draw(bg)
    for center in agent_pix.cpu().numpy():
        for cen in center:
            c0, c1 = round(cen[1]), round(cen[0])
            draw.ellipse((c0-circle_rad, c1-circle_rad, c0+circle_rad, c1+circle_rad), fill = 'red', outline='red')
        
    return bg

def overlay_goals_agents(bg, obj, goal, agent, circle_rad:float=5):
    bg = bg.copy()
    W, H = bg.size
    for i in range(len(obj)):
        obj[i] = obj[i].resize((W // 5, H // 5), Image.LANCZOS)
        
    agent = agent.squeeze(1)
    
    goals_pix = ((1+goal)/2 * torch.tensor([[H, W]]*goal.size(0), device=goal.device, dtype=goal.dtype))
    agent_pix = ((1+agent)/2 * torch.tensor([[H, W]]*agent.size(0), device=agent.device, dtype=agent.dtype))

    for i, center in enumerate(goals_pix.cpu().numpy()):
        c0, c1 = round(center[1]),round(center[0])
        w, h = obj[i].size
        bg.paste(obj[i], (c0 - w//2, c1 - h//2), obj[i])
        
    draw = ImageDraw.Draw(bg)
    
    for center in agent_pix.cpu().numpy():
        c0, c1 = round(center[1]), round(center[0])
        draw.ellipse((c0-circle_rad, c1-circle_rad, c0+circle_rad, c1+circle_rad), fill = 'red', outline='red')
        
    return bg