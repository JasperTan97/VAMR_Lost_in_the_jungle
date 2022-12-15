#!/bin/bash

conda create -n vamr_proj python==3.10
conda activate vamr_proj
pip3 install --user -r requirements.txt