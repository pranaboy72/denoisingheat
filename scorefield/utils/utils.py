import os
import einops
import matplotlib.pyplot as plt

def log_num_check(path):
    log_num = 0
    while True:
        new_path = path + '_' + str(log_num)
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