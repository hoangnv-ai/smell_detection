# #!/bin/bash

SMELL_NAME=$1
cd /workspace/smell_detection
python train/prompt_tuning.py --train_file /workspace/smell_detection/dataset/${SMELL_NAME}/train.csv --valid_file /workspace/smell_detection/dataset/${SMELL_NAME}/val.csv --model_name_or_path "deepseek-ai/deepseek-coder-1.3b-base"

