import imageio
import numpy as np
import matplotlib.pyplot as plt

def slice_to_rgb(slice_):
    # Using the imshow properties to generate a visual appearance
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(slice_, cmap='hot', interpolation='nearest', origin='lower', alpha=0.5)
    ax.axis('off')  # turn off axis for cleaner video frames

    # Extract the image content from the figure
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # close the figure to free up memory

    return image_from_plot

imgs = np.load('../results/heat/eval.npy', allow_pickle=True)
rgb_frames = [slice_to_rgb(slice_) for slice_ in imgs]

video_filename = 'eval.mp4'

fps = 20
with imageio.get_writer(video_filename, mode='I', fps=fps) as writer:
    for frame in rgb_frames:
        writer.append_data(frame)
        
        