import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
import gym
import warnings

from scorefield.datasets.d4rl import load_environment

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))


MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12),
    'maze2d-open-v2': (0, 8, 0, 8)
}

class MazeRenderer:
    def __init__(self, env):
        if type(env) is str: env = load_environment(env)
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
    def __init__(self, env):   # Unlike Diffuser, we use birdeye-view of maze env with pixel image 
        self.env_name = env
        self.env = load_environment(env)
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
        
    def renders(self, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]
        
        if len(bounds) == 2:
            _, scale = bounds
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
        
        return super().renders(**kwargs)
            