# LWIRcolorization
Infrared image colorization based on full-scale connection and feature alignment
We provide the Pytorch implementation of "Infrared image colorization based on full-scale connection and feature alignment"
# Getting Started
## Installation
This code was tested with Pytorch 1.7.0, CUDA 10.2, and Python 3.6
## Testing
* Please download the pre-trained model and test set from [here](https://drive.google.com/drive/folders/1vVRhSVFs6yt2R-17dTURf5NeNzAscvkk?usp=sharing), and put model under ./checkpoints/experiment_name/ . put test set under ./datasets/
* test the model 
 ```python test.py --dataroot datasets/cyclegan --model sc ```
 # Acknowledge
 Our code is developed based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation)
