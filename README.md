# Denoising Heat-inspired Diffusion

## Installation
    conda create -n dnheat python=3.9
    conda activate dnheat
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install tensorboard
    pip install -r requirements.txt

## Download pretrained model
From https://drive.google.com/drive/folders/1HGux_54jAf3KYAAvYMKWrSpJsG267HSd?usp=drive_link, you can download .pt file for the weights and .yaml for the parameters.

## Reference
    @article{chang2023denoising,
      title={Denoising Heat-inspired Diffusion with Insulators for 
    Collision Free Motion Planning},
      author={Chang, Junwoo and Ryu, Hyunwoo and Kim, Jiwoo and Yoo, Soochul and Seo, Joohwan and Prakash, Nikhil and Choi, Jongeun and Horowitz, Roberto},
      journal={arXiv preprint arXiv:2310.12609},
      year={2023}
    }
