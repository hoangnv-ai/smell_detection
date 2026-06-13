import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftType, PromptTuningConfig, get_peft_model
# Thêm classification_report để tính toán đầy đủ Precision, Recall, F1
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Thêm đường dẫn utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *

# ==========================================
# 1. CẤU HÌNH ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser("Prompt Tuning for Code Smell Detection")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
parser.add_argument("--train_file", type=str, default="/workspace/PEFT4CSD/dataset/RQ1/CC/train.csv")
parser.add_argument("--valid_file", type=str, default="/workspace/PEFT4CSD/dataset/RQ1/CC/valid.csv")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_train_samples", type=int, choices=[100, 200, 500, 1000], default=None)
parser.add_argument("--num_virtual_tokens", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--should_log", type=bool, default=True)
parser.add_argument("--output_dir", type=str, default="output")

# Cấu hình Dataset gốc (DeepSmells config)
parser.add_argument("--smell", type=str, default="MultifacetedAbstraction", choices=['ComplexMethod', 'ComplexConditional', 'FeatureEnvy', 'MultifacetedAbstraction'])
parser.add_argument("--dim", type=str, default="1d")
parser.add_argument("--fast_dev_run", type=bool, default=True)
args = parser.parse_args()

# ==========================================
# 2. SETUP LOGGING & SEED
# ==========================================
model_name = args.model_name_or_path.split("/")[-1]
log_dir = os.path.join(args.output_dir, model_name, f"prompt_tuning_seed_{args.seed}")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(log_dir, "train_log.txt"))],
)
logging.getLogger().handlers[0].setLevel(logging.WARNING)
logger.setLevel(logging.INFO if args.should_log else logging.WARN)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# In thông tin cấu hình
config_profile = (
    f"{'='*50}\nPrompt Tuning Configuration\n"
    f"Seed: {args.seed} | Model: {args.model_name_or_path}\n"
    f"Smell: {args.smell} | Dim: {args.dim} | Fast Dev Run: {args.fast_dev_run}\n"
    f"Epochs: {args.num_epochs} | Batch Size: {args.batch_size} | LR: {args.learning_rate:.0e}\n"
    f"Virtual Tokens: {args.num_virtual_tokens} | Optimizer: {args.optimizer}\n{'='*50}\n"
)
print(config_profile)
logger.info(config_profile)

# ==========================================
# 3. XỬ LÝ DỮ LIỆU & DATASET
# ==========================================
@dataclass
class InputData:
    train_data: np.ndarray
    train_labels: np.ndarray
    eval_data: np.ndarray
    eval_labels: np.ndarray
    max_input_length: int

def count_lines(folder: Path) -> int:
    total = 0
    for file in folder.glob("*"):
        if file.is_file():
            with file.open("r", errors="ignore") as f:
                total += sum(1 for _ in f)
    return total

def read_token_lengths(folder: Path, is_c2v: bool = False):
    lengths = []
    dtype = np.float32 if is_c2v else np.int32
    for file in folder.glob("*"):
        if file.name.startswith(".") or not file.is_file():
            continue
        with file.open("r", errors="ignore") as f:
            for line in f:
                text = line.replace("\t", " ").strip()
                if text:
                    arr = np.fromstring(text, dtype=dtype, sep=" ")
                    if len(arr) > 0:
                        lengths.append(len(arr))
    return lengths

def compute_max_without_upper_outliers(lengths, z: float = 1.0) -> int:
    if not lengths:
        return 0
    values = np.asarray(lengths, dtype=np.float64)
    cutoff = values.mean() + z * values.std()
    filtered = values[values <= cutoff]
    return int(filtered.max()) if len(filtered) > 0 else 0

def get_outlier_threshold(data_path: Path, z: float = 1.0, is_c2v: bool = False) -> int:
    pos_threshold = compute_max_without_upper_outliers(read_token_lengths(data_path / "Positive", is_c2v), z)
    neg_threshold = compute_max_without_upper_outliers(read_token_lengths(data_path / "Negative", is_c2v), z)
    return max(pos_threshold, neg_threshold)

def retrieve_data(folder: Path, max_len: int, is_c2v: bool = False):
    samples = []
    dtype = np.float32 if is_c2v else np.int32
    for file in folder.glob("*"):
        if file.name.startswith(".") or not file.is_file():
            continue
        with file.open("r", errors="ignore") as f:
            for line in f:
                text = line.replace("\t", " ").strip()
                if not text:
                    continue
                arr = np.fromstring(text, dtype=dtype, sep=" ")
                if 0 < len(arr) <= max_len:
                    padded = np.zeros(max_len, dtype=np.float32)
                    padded[:len(arr)] = arr
                    samples.append(padded)
    return samples

def get_data(data_path, 
            train_validate_ratio=0.7, 
            max_training_samples=5000, 
            max_eval_samples=150000, 
            is_c2v=False, 
            seed=0):
    data_path = Path(data_path)
    rng = random.Random(seed)

    max_input_length = get_outlier_threshold(data_path, z=1, is_c2v=is_c2v)
    if max_input_length <= 0:
        raise ValueError(f"Không tìm thấy sample hợp lệ trong {data_path}")

    pos_data = retrieve_data(data_path / "Positive", max_input_length, is_c2v=is_c2v)
    neg_data = retrieve_data(data_path / "Negative", max_input_length, is_c2v=is_c2v)
    rng.shuffle(pos_data)
    rng.shuffle(neg_data)

    train_pos = int(train_validate_ratio * len(pos_data))
    eval_pos = len(pos_data) - train_pos
    train_neg = int(train_validate_ratio * len(neg_data))
    eval_neg = len(neg_data) - train_neg

    train_pos = train_neg = min(max_training_samples, train_pos, train_neg)

    if max_eval_samples is not None and eval_neg > max_eval_samples:
        removed_ratio = (eval_neg - max_eval_samples) / eval_neg
        eval_pos = int(eval_pos - eval_pos * removed_ratio)
        eval_neg = max_eval_samples

    training_data = pos_data[:train_pos] + neg_data[:train_neg]
    training_labels = np.zeros(len(training_data), dtype=np.float32)
    training_labels[:train_pos] = 1.0

    eval_data = (pos_data[-eval_pos:] if eval_pos > 0 else []) + (neg_data[-eval_neg:] if eval_neg > 0 else [])
    eval_labels = np.zeros(len(eval_data), dtype=np.float32)
    eval_labels[:eval_pos] = 1.0

    training_data = np.array(training_data, dtype=np.float32)
    eval_data = np.array(eval_data, dtype=np.float32)

    train_perm = np.random.default_rng(seed).permutation(len(training_labels))
    eval_perm = np.random.default_rng(seed + 1).permutation(len(eval_labels))
    
    return training_data[train_perm], training_labels[train_perm], eval_data[eval_perm], eval_labels[eval_perm], max_input_length

def get_all_data(data_root, 
                smell, 
                dim="1d", 
                train_validate_ratio=0.7, 
                max_training_samples=5000, 
                max_eval_samples=None, 
                seed=0):
    if max_eval_samples is None:
        max_eval_samples = 150000 if smell in ["ComplexConditional", "ComplexMethod"] else 50000

    data_path = Path(data_root) / smell / dim
    train_data, train_labels, eval_data, eval_labels, max_input_length = get_data(
        data_path, train_validate_ratio, max_training_samples, max_eval_samples, seed=seed
    )

    all_data = np.concatenate((train_data, eval_data), axis=0)
    all_labels = np.concatenate((train_labels, eval_labels), axis=0)

    train_data, eval_data, train_labels, eval_labels = train_test_split(
        all_data, all_labels, test_size=(1.0 - train_validate_ratio), stratify=all_labels, random_state=seed
    )

    return InputData(train_data, train_labels, eval_data, eval_labels, max_input_length)

DATA_ROOT = Path("/workspace/PEFT4CSD/dataset/data/tokenizer_cs")
max_eval_limit = 5000 if args.fast_dev_run else None

input_data = get_all_data(
    DATA_ROOT, 
    smell=args.smell, 
    dim=args.dim, 
    max_training_samples=5000, 
    max_eval_samples=max_eval_limit, 
    seed=args.seed
)

class CodeSmellDataset(Dataset):
    def __init__(self, inputs, labels=None, is_test=False):
        self.inputs = inputs
        self.labels = labels
        self.is_test = is_test

    def __getitem__(self, idx):
        sample_input = torch.tensor(self.inputs[idx].reshape(1, -1), dtype=torch.float32)
        if self.is_test:
            return sample_input
        sample_label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return sample_input, sample_label

    def __len__(self):
        return len(self.inputs)

train_set = CodeSmellDataset(input_data.train_data, input_data.train_labels)
valid_set = CodeSmellDataset(input_data.eval_data, input_data.eval_labels)

train_dataloader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=4,       # số CPU cores để prefetch
    pin_memory=True,     # transfer CPU→GPU nhanh hơn
    persistent_workers=True  # tránh spawn process mỗi epoch
)
valid_dataloader = DataLoader(
    valid_set, batch_size=args.batch_size, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# ==========================================
# 4. KHỞI TẠO TOKENIZER, MODEL & PEFT
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.num_virtual_tokens)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
logger.info(f"Prompt Tuning-Trainable parameters: {model.get_nb_trainable_parameters()}")

if "deepseek" in args.model_name_or_path or "starcoder" in args.model_name_or_path:
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id

if torch.cuda.is_available():
    model.cuda()

optimizer = AdamW(model.parameters(), lr=args.learning_rate) if args.optimizer.lower() == "adamw" else None
num_training_steps = len(train_dataloader) * args.num_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(0.06 * num_training_steps),
    num_training_steps=num_training_steps
)

# ==========================================
# 5. TIÊN TRÌNH HUẤN LUYỆN (TRAINING & VALIDATION)
# ==========================================
best_validation_loss = float("inf")
peak_memory = 0
start_time = time.time()




for epoch in range(args.num_epochs):
    # --- TRAINING LAYER ---
    model.train()
    train_loss = 0.0
    
    progress_bar_train = tqdm(
        train_dataloader, 
        desc=f"Training Epoch {epoch + 1}/{args.num_epochs}", 
        leave=True
    )

    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    for step, batch in enumerate(progress_bar_train):
        input_ids = batch[0].squeeze(1).long().cuda()
        labels = batch[1].squeeze(1).long().cuda()

        with autocast(dtype=torch.bfloat16):  # hoặc float16 nếu GPU cũ
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar_train.set_postfix({"loss": f"{loss.item():.4f}"})

        current_memory = torch.cuda.max_memory_allocated()
        if current_memory > peak_memory:
            peak_memory = current_memory

    avg_train_loss = train_loss / len(train_dataloader)
    logger.info(f"Epoch {epoch + 1} - Avg Training loss: {avg_train_loss:.4f}")
    print(f"Epoch {epoch + 1} - Avg Training loss: {avg_train_loss:.4f}")

    # --- VALIDATION LAYER ---
    model.eval()
    total_validation_loss = 0.0
    
    # Khởi tạo danh sách để lưu tất cả ground-truths và predictions
    all_preds = []
    all_labels = []

    progress_bar_valid = tqdm(
        valid_dataloader, 
        desc=f"Validation Epoch {epoch + 1}/{args.num_epochs}", 
        leave=True
    )

    for batch in progress_bar_valid:
        input_ids = batch[0].squeeze(1).long().cuda()
        labels = batch[1].squeeze(1).long().cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_validation_loss += loss.item()

            # Lấy Logits từ Model output ra để tính nhãn dự đoán cao nhất (Argmax)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Đẩy dữ liệu tensor từ GPU về CPU và chuyển sang numpy array để sklearn xử lý
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_validation_loss = total_validation_loss / len(valid_dataloader)
    logger.info(f"Epoch {epoch + 1} - Avg Validation loss: {avg_validation_loss:.4f}")
    print(f"Epoch {epoch + 1} - Avg Validation loss: {avg_validation_loss:.4f}")

    # --- TÍNH TOÁN VÀ IN RA PRECISION, RECALL, F1-SCORE ---
    # Sử dụng classification_report của sklearn để tính toán toàn bộ chỉ số
    metrics_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=["Negative", "Positive"], 
        digits=4,
        zero_division=0
    )
    
    # In ra terminal và ghi vào log file text cùng lúc với việc checkpoint
    print(f"\n--- Validation Metrics Report for Epoch {epoch + 1} ---")
    print(metrics_report)
    logger.info(f"\nValidation Metrics Report - Epoch {epoch + 1}:\n{metrics_report}")

    # Lưu checkpoint tốt nhất (Best model dựa theo Loss thấp nhất)
    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        best_model_path = os.path.join(log_dir, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)
        
        # Ghi chú lại report của model tốt nhất vào file riêng nếu cần thiết
        with open(os.path.join(best_model_path, "best_metrics_report.txt"), "w") as f_rep:
            f_rep.write(metrics_report)

    # Lưu checkpoint định kỳ của Epoch hiện tại
    epoch_save_path = os.path.join(log_dir, f"epoch_{epoch + 1}")
    os.makedirs(epoch_save_path, exist_ok=True)
    model.save_pretrained(epoch_save_path)

# ==========================================
# 6. LƯU REPORT KẾT QUẢ CUỐI CÙNG
# ==========================================
end_time = time.time()
training_time = end_time - start_time

os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)

with open(f"{args.output_dir}/{model_name}/peak_memory.txt", "a") as f:
    f.write(f"prompt tuning: {peak_memory}\n")

with open(f"{args.output_dir}/{model_name}/training_time.txt", "a") as f:
    f.write(f"epoch: {args.num_epochs} | prompt tuning time: {training_time:.2f}s\n")

print(f"\n[DONE] Hoàn thành huấn luyện trong {training_time:.2f} giây. Peak Memory: {peak_memory / (1024**2):.2f} MB")