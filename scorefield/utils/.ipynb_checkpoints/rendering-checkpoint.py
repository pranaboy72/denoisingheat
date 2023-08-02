import os
import numpy as np
import torch
import einops
import imageio
import matplotlib.pyplot as plt
import gym
import warnings
# import d4rl
import cv2
import pandas as pd
import random

# from scorefield.datasets.d4rl import load_environment
from scorefield.utils.utils import save_obs, random_rgb, get_distractors

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    img = np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))[:,:,:3]
    einops.rearrange(img, 'h w c -> c h w')
    return img


MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12),
    'maze2d-open-v1': (0, 8, 0, 8)
}

class MazeRenderer:
    def __init__(self, env):
#         if type(env) is str: env = load_environment(env)
        self._config = env._config
        self._background = self._config != ' '
        
        
    def renders(self, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
                   extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)
        plt.axis('off')
        plt.title(title)
        
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img
    

class Maze2dRenderer(MazeRenderer):
    def __init__(self, args): 
#         self.env_name = args['env_name']
#         self.env = load_environment(args['env_name'])
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
        self.img_size = args['image_size']
        self.dists = args['distractor_num']
        
        self.goal_col = [255, 0, 0]
        self.agent_col = [0, 0, 255]
        
    def map_init(self):
        self.map = super().renders() # Get the map img
        self.init_x, self.init_y, self.block_x, self.block_y = self.search_init_block(self.map)
        self.r = 0
        self.g = 0
        self.b = 255
        
        self.bounds = MAZE_BOUNDS[self.env_name] # (0, x, 0, x)
        
        self.distractors = get_distractors(self.env._target, self.bounds, self.dists)
        
        # self.map_img = self.stamp(self.map, self.env._target, 'goal') # stamp the goal position
        # self.map_img = self.stamp(self.map_img, self.distractors, 'distractors') # stamp the distractors
        return self.map
        
    def renders(self, save=False):        
        img = self.stamp(self.map, self.env._target, 'goal')   # goal
        # img = self.stamp(img, self.env.sim.data.qpos, 'agent')  # agent
        if save: save_obs(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        img = einops.rearrange(img, 'h w c -> c h w')
        
        return img
            
    def search_init_block(self,img):
        init_found = False
        block_found = False
        for i in range(img[...,0].shape[0]):
            for j in range(img[...,1].shape[1]):
                if img[i,j,0] != 255 and init_found is False:
                    init_x = i
                    init_y = j
                    init_found = True
                if img[i-1,j,0] != 255 and img[i,j-1,0] != 255 and img[i,j,0] == 255 and init_found:
                    block_x = i - init_x
                    block_y = j - init_y
                if block_found: break
            if block_found: break
        return init_x, init_y, block_x, block_y
    
    def change_colors(self):
        if self.b > 0:
            self.g = min(self.g + 3, 255)
            self.b = max(self.b - 3, 0)
        elif self.r < 255:
            self.r = min(self.r + 3, 255) 
        else:
            self.g = max(self.g - 3, 0)
        
        return [self.r, self.g, self.b]
            
    
    def stamp(self, img, aim, col, stamp_size=10):   
        target_point = aim.copy()
        
        
        if col == 'goal': 
            # col = [255, 0, 255]
            col = self.goal_col
        elif col == 'agent': 
            # col = self.change_colors() # To make the trajectory of the agent
            col = self.agent_col
        elif col == 'distractors':
            cols = random_rgb(self.goal_col, len(target_point))


        if isinstance(target_point, list):
            target_point = np.array([target_point])
    
        if len(target_point.shape) == 1: 
            target_point = np.reshape(target_point, (1,-1))
        elif len(target_point.shape) == 3:
            target_point = np.reshape(target_point, (-1,2))
        
        target_point[:,0], target_point[:,1] = self.init_x + self.block_x + target_point[:,0] * self.block_x // 2,\
            self.init_y + self.block_y + target_point[:,1] * self.block_y // 2
            
        
        # for i, point in enumerate(target_point):  
            # Get bounds of the stamp region
        for x, y in target_point:
            start_x = int(x - stamp_size // 2)
            end_x =  int(x + stamp_size // 2)
            start_y =  int(y - stamp_size // 2)
            end_y = int(y + stamp_size // 2)
            
                # Make stamps
                # if isinstance(col, list):
            img[start_x:end_x, start_y:end_y, :] = col
            # else:
            #     img[start_x:end_x, start_y:end_y, :] = cols[i]
            
        return img
