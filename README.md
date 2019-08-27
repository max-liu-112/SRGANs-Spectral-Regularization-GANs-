# SRGANs
## Implementation of Spectral Regularization GANs
### References
-Kanglin Liu, Wenming Tang, Fei Zhou, Guoping Qiu. *Spectral Regularization for Combating Mode Collapse in GANs*. ICCV2019. [SRGANs]
## Setup

### Install required python libraries:
chainer==3.3.0
tensorflow-gpu==1.2.0
numpy==1.11.1
cython==0.27.2
cupy==2.0.0
scipy==0.19.0
pillow==4.3.0
pyyaml==3.12
h5py==2.7.1

### Download inception model: 

`python source/inception/download.py --outfile=datasets/inception_model`

### Spectral Collapse VS Mode Collapse

<img src="https://github.com/max-liu-112/SRGANs/blob/master/Images/fig1.png">

<img src="https://github.com/max-liu-112/SRGANs/blob/master/Images/fig2.png">

<img src="https://github.com/max-liu-112/SRGANs/blob/master/Images/fig3.png">

<img src="https://github.com/max-liu-112/SRGANs/blob/master/Images/fig4.png">
