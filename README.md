# Denoising Heat-inspired Diffusion

## Installation
    conda create -n dnheat python=3.9
    conda activate dnheat
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install tensorboard
    pip install -r requirements.txt

## Using pretrained model
Please download the 'runs' file from [this link](https://drive.google.com/drive/folders/1nskuIuQHy8V4m4Nzd2sRnJiKaXfma1Nm?usp=drive_link) and incorporate it into the repository.

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
