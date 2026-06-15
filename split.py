import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser("")
parser.add_argument("--name_folder", type=str)
args = parser.parse_args()

for name_folder in [args.name_folder]:
    print("-"*100)
    base_dir = f"/workspace/DeepLearningSmells/dataset/{name_folder}"

    # Đọc dữ liệu
    df = pd.read_csv(f"{base_dir}/all.csv")
    print(df["label"].value_counts())

    # Bước 1: Tách train (60%) và phần còn lại (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # Bước 2: Tách phần còn lại thành val (20%) và test (20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    # Lưu file
    train_df.to_csv(f"{base_dir}/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/val.csv", index=False)
    test_df.to_csv(f"{base_dir}/test.csv", index=False)

    # Kiểm tra số lượng
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    # Kiểm tra tỷ lệ nhãn
    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nVal label distribution:")
    print(val_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    break
