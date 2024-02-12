# Denoising Heat-inspired Diffusion
Training and visualization of the model from [Denoising Heat-inspired Diffusion with Insulators for Collision Free Motion Planning](https://sites.google.com/view/denoising-heat-inspired?usp=sharing) (Neurips 2023 Workshop on Diffusion Models).

<p align="center"><img src="https://github.com/pranaboy72/denoisingheat/assets/86711384/c5fc8259-abd6-4b5e-8c65-99370fa85fbf" width="40%" height="40%"/>  
<img src="https://github.com/pranaboy72/denoisingheat/assets/86711384/36053f66-a2b9-4aed-b363-02b471242c48" width="40%" height="40%"/>


## Installation
    conda create -n dnheat python=3.9
    conda activate dnheat
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install tensorboard
    pip install -r requirements.txt

## Using pretrained model
Please download the 'runs' file from [this link](https://drive.google.com/drive/folders/1nskuIuQHy8V4m4Nzd2sRnJiKaXfma1Nm?usp=drive_link) and incorporate it into the repository.

## Inference
Evaluate the model in a random or pre-set map, as specified in `inference.ipynb`, applying both the pre-trained weights and the hyperparameters.   


## Training from scratch
1. Train your own model using `train.ipynb`
   * Default hyperparameters are located in `./denoisingheat/configs/heat_diffusion.yaml`.
   * You can modify the hyperparameters in the `heat_diffusion.yaml` file.
   
2. Evaluate your trained model using `inference.ipynb`
    * Update the 'config_dir' path in the first block and the '.pt' file path in the second block to reflect the directory where you saved your new model.

3. Monitor the training progress and analyze logs via TensorBoard:

```
tensorboard --logdir=./runs
```


## Reference
    @article{chang2023denoising,
      title={Denoising Heat-inspired Diffusion with Insulators for 
    Collision Free Motion Planning},
      author={Chang, Junwoo and Ryu, Hyunwoo and Kim, Jiwoo and Yoo, Soochul and Seo, Joohwan and Prakash, Nikhil and Choi, Jongeun and Horowitz, Roberto},
      journal={arXiv preprint arXiv:2310.12609},
      year={2023}
    }

## Acknowledgement
Our diffusion model is implemented based on Phil Wang's ['denoising-diffusion-pytorch'](https://github.com/lucidrains/denoising-diffusion-pytorch) GitHub repository.
