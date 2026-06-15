import argparse, os, sys, time, torch, logging
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, PeftType, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, PeftConfig, PeftModel)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed)
from tqdm import tqdm
from writer import *

parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
parser.add_argument("--train_file", type=str, default="/workspace/DeepLearningSmells/dataset/ComplexConditional/train.csv")
parser.add_argument("--valid_file", type=str, default="/workspace/DeepLearningSmells/dataset/ComplexConditional/val.csv")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--max_train_samples", type=int, choices=[100, 200, 500, 1000], default=None)
parser.add_argument("--num_virtual_tokens", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=3e-4) 
parser.add_argument("--optimizer", type=str, default="Adamw")
parser.add_argument("--should_log", type=bool, default=True)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--eval_steps", type=int, default=500)  # <-- thêm arg mới
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

model_name = args.model_name_or_path.split("/")[-1]

############################################################################################################
# Setup logging
logger = logging.getLogger(__name__)
log_file_path = os.path.join(args.output_dir, model_name, f"prompt_tuning_seed_{args.seed}", f"train_max_samples_{args.max_train_samples}_log.txt")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)

logging.getLogger().handlers[0].setLevel(logging.WARNING)
logger.setLevel(logging.INFO if args.should_log else logging.WARN)

content_write = gen_content_write(args)
print(content_write)
logger.info(content_write)

set_seed(args.seed)
use_cuda = True

############################################################################################################
# Set peft config
peft_type = PeftType.PROMPT_TUNING

peft_config = PromptTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=args.num_virtual_tokens
)

if "codet5" in args.model_name_or_path.lower():
    logging.warning("CodeT5 models are not supported yet. Please use another model.")
    sys.exit(1)

############################################################################################################
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

############################################################################################################
# Load datasets
train_df = load_trainset(args.train_file, max_train_samples=args.max_train_samples, seed=args.seed)
eval_df = pd.read_csv(args.valid_file)

trainset = Dataset.from_pandas(train_df)
evalset = Dataset.from_pandas(eval_df)

datasets = DatasetDict({
    'train': trainset,
    'validation': evalset
})

label_list = train_df["label"].unique()
num_labels = len(label_list)

############################################################################################################
# Tokenize datasets
if args.max_seq_length > tokenizer.model_max_length:
    logging.warning(
        f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

if "bert" or "unixcoder" in args.model_name_or_path.lower():
    tokenize_max_length = max_seq_length - args.num_virtual_tokens
else:
    tokenize_max_length = max_seq_length

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=tokenize_max_length)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "text_label"],
    load_from_cache_file=False
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    outputs = tokenizer.pad(examples, return_tensors="pt", padding=True, max_length=max_seq_length)
    return outputs

train_dataloader = DataLoader(
    tokenized_datasets["train"], batch_size=args.batch_size, 
    shuffle=True, collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=args.batch_size, 
    shuffle=False, collate_fn=collate_fn
)

############################################################################################################
# Load model
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
logger.info(f"Prompt Tuning-Trainable parameters: {model.get_nb_trainable_parameters()}")

if "deepseek" or "starcoder" in args.model_name_or_path:
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

if args.optimizer.lower() == "adamw":
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs)
)

total_steps = 0
best_validation_loss = float("inf")
peak_memory = 0
if use_cuda:
    model.cuda()

############################################################################################################
# Hàm eval tách riêng để tái sử dụng
def run_evaluation(model, valid_dataloader, use_cuda):
    model.eval()
    total_validation_loss = 0.0

    progress_bar_valid = tqdm(
        total=len(valid_dataloader),
        desc="Evaluating",
        position=0,
        mininterval=1,
        leave=True
    )

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()} if use_cuda else batch
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_validation_loss += loss.item()

        if step % 5 == 0:
            progress_bar_valid.update(5)

    progress_bar_valid.close()
    avg_val_loss = total_validation_loss / len(valid_dataloader)
    return avg_val_loss

############################################################################################################
# Training
# Checkpoint dùng chung 1 path, overwrite để tiết kiệm bộ nhớ
latest_checkpoint_path = os.path.join(args.output_dir, model_name, f"prompt_tuning_seed_{args.seed}", "latest_checkpoint")
best_model_path = os.path.join(args.output_dir, model_name, f"prompt_tuning_seed_{args.seed}", "best_model")
os.makedirs(latest_checkpoint_path, exist_ok=True)
os.makedirs(best_model_path, exist_ok=True)

start_time = time.time()

for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0.0

    progress_bar_train = tqdm(
        total=len(train_dataloader), 
        desc=f"Training epoch {epoch + 1}",
        position=0,
        mininterval=1,
        leave=True
    )

    for step, batch in enumerate(train_dataloader):
        total_steps += 1
        batch = {k: v.cuda() for k, v in batch.items()} if use_cuda else batch
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            progress_bar_train.set_postfix({"loss": loss.item(), "global_step": total_steps})
            progress_bar_train.update(5)

        current_memory = torch.cuda.max_memory_allocated()
        if current_memory > peak_memory:
            peak_memory = current_memory


    avg_val_loss = run_evaluation(model, valid_dataloader, use_cuda)

    log_msg = (
        f"[Epoch {epoch + 1} | "
        f"Train loss: {loss.item():.4f} | "
        f"Val loss: {avg_val_loss:.4f}"
    )
    logger.info(log_msg)
    print(log_msg)

    # Lưu latest checkpoint (overwrite để tiết kiệm bộ nhớ)
    model.save_pretrained(latest_checkpoint_path)
    logger.info(f"[Step {total_steps}] Latest checkpoint saved (overwrite): {latest_checkpoint_path}")

    # Lưu best model nếu val loss tốt hơn
    if avg_val_loss < best_validation_loss:
        best_validation_loss = avg_val_loss
        model.save_pretrained(best_model_path)
        logger.info(f"[Step {total_steps}] ✅ New best model saved (val_loss={avg_val_loss:.4f}): {best_model_path}")

    # Quay lại train mode
    model.train()

progress_bar_train.close()

avg_train_loss = train_loss / len(train_dataloader)
logger.info(f"Epoch {epoch + 1} - Avg Training loss: {avg_train_loss:.4f}")
print(f"Epoch {epoch + 1} - Avg Training loss: {avg_train_loss:.4f}")

############################################################################################################
# Save peak memory & training time
with open(f"{args.output_dir}/{model_name}/peak_memory.txt", "a") as f:
    f.write(f"prompt tuning: {str(peak_memory)}\n")

end_time = time.time()
training_time = end_time - start_time

with open(f"{args.output_dir}/{model_name}/training_time.txt", "a") as f:
    f.write(f"epoch: {args.num_epochs} prompt tuning: {str(training_time)}\n")
