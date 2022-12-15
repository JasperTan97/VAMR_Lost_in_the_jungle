#!/bin/bash

# Call this program from the root directory of the git repo.

conda create -n vamr_proj python==3.10
conda activate vamr_proj
pip3 install --user -r requirements.txt

# Download datasets
cd data
wget https://rpg.ifi.uzh.ch/docs/teaching/2022/parking.zip
wget https://rpg.ifi.uzh.ch/docs/teaching/2022/kitti05.zip
wget https://rpg.ifi.uzh.ch/docs/teaching/2022/malaga-urban-dataset-extract-07.zip

# unzip datasets
unzip -a parking.zip
unzip -a kitti05.zip
unzip -a malaga-urban-dataset-extract-07.zip

# return to the original directory.
cd ..