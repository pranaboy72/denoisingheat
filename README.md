# Denoising Heat-inspired Diffusion

## Installation
    conda create -n denoisingheat python=3.9
    conda activate denoisingheat
    pip install -r requirements.txt

## Download pretrained model
From https://drive.google.com/drive/folders/1HGux_54jAf3KYAAvYMKWrSpJsG267HSd?usp=drive_link, you can download .pt file for the weights and .yaml for the parameters.


## Evaluation
To evaluate the model, check out inference.ipynb.   
### First cell
Load the pretrained model and its corresponding parameters.
### Second cell
Generate the obstacle(s) on the background.   
### Third/Fourth/Fifth cell
Generate additional obstacle(s) for the unreachable goal, and decide the initial distribution location.
### Sixth cell
Set the parameters for Langevin Dynamics, goal location. You can check out each episode(each row)'s result through interval images. 
### Last cell
You can save all the images to make a video of the result.
   
## Training
You can train your model by changing the parameters in denoisingheat/configs/heat_diffusion.yaml and running trainer.ipynb.   
Monitor the training with tensorboard:
    tensorboard --logdir=./runs

## Reference
    @article{chang2023denoising,
      title={Denoising Heat-inspired Diffusion with Insulators for 
    Collision Free Motion Planning},
      author={Chang, Junwoo and Ryu, Hyunwoo and Kim, Jiwoo and Yoo, Soochul and Seo, Joohwan and Prakash, Nikhil and Choi, Jongeun and Horowitz, Roberto},
      journal={arXiv preprint arXiv:2310.12609},
      year={2023}
    }
