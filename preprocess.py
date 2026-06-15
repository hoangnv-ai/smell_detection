import os
import pandas as pd
import tqdm

import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--name_folder", type=str)
parser.add_argument("--rate", type=int)
args = parser.parse_args()

print("args.name_folder :", args.name_folder)
base_folder = "/workspace/DeepLearningSmells/data/training_data_cs"
base_output_folder = "/workspace/DeepLearningSmells/dataset"
if not os.path.exists(base_output_folder):
    os.mkdir(base_output_folder)


dict_label = {"Negative": 0, "Positive": 1}

rate_neg = {
    "ComplexConditional" : 20, 
    "ComplexMethod" : 10, 
    "FeatureEnvy" : 10, 
    "MultifacetedAbstraction" : 10, 
}
for name_folder in [args.name_folder]:
    rows = []
    count = 0
    for label_txt in list(dict_label.keys()):
        path_pos_dir = os.path.join(base_folder, name_folder, label_txt)
        list_pos_file = os.listdir(path_pos_dir)

        print(f"{name_folder}")
        print(f"label: {label_txt}")
        for file in tqdm.tqdm(list_pos_file):
            file_path = os.path.join(path_pos_dir, file)
            # print("file_path :", file_path)
            with open(file_path, "r") as f:
                text = f.read()
            
            label = dict_label[label_txt]

            if label == 0:
                count += 1
                if not ((count % args.rate) == 0):
                    continue

            text_label = name_folder
            # print("text: ", text)
            # print("label :", label)
            # print("text_label :", text_label)
            row = {
                "text": text,
                "label": label,
                "text_label": text_label,
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    type_data = "all"
    base_output_subfolder = os.path.join(base_output_folder, name_folder)
    if not os.path.exists(base_output_subfolder):
        os.mkdir(base_output_subfolder)
    path_file = os.path.join(base_output_subfolder, f"{type_data}.csv")
    df.to_csv(path_file, 
            index=False, 
            encoding="utf-8-sig")

    break
