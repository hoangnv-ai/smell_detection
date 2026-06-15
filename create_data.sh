cd
git clone https://github.com/tushartushar/DeepLearningSmells.git
mv /root/DeepLearningSmells/data/training_data_cs/ComplexConditional.7z /workspace/DeepLearningSmells/data/training_data_cs
cd /workspace/DeepLearningSmells/data/training_data_cs
7z x ComplexConditional.7z
rm -rf ComplexConditional.7z
rm -rf /root/DeepLearningSmells
cd /workspace
python preprocess.py
python split.py #phải sửa trong file
