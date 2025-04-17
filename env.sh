cd /userhome/2072/fyp24093/FYP
#/userhome/2072/fyp24093/anaconda3/bin/conda create -p ./condaenv python=3.11
source /userhome/2072/fyp24093/anaconda3/bin/activate ./condaenv

#conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia
#pip install git+https://github.com/end2you/end2you.git
#pip install kagglehub
#pip install pandas
#pip install natsort
#pip install opencv-python
#pip install librosa
#conda install -c conda-forge ffmpeg



python train.py 1>3.log 2>4.log &