#!/bin/bash

SMELL_NAME=$1

cd /workspace

git clone https://github.com/tushartushar/DeepLearningSmells.git

sudo apt update
sudo apt install p7zip-full -y

cd /workspace/DeepLearningSmells/data/training_data_cs

7z x "${SMELL_NAME}.7z"

cd /workspace/smell_detection

python preprocess.py --name_folder "${SMELL_NAME}" --rate 20
python split.py --name_folder "${SMELL_NAME}"

mv /workspace/DeepLearningSmells/dataset/ComplexMethod /workspace/smell_detection/dataset
