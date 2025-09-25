#@title Load Dataset and Model

from datasets import load_dataset
from transformers import AutoTokenizer

def get_train_val_split(model_name, dataset_name="sst2", sample_data=True,):
    print("Getting Train and Test splits.")
    train_split=500
    eval_split=200
    dataset_raw = load_dataset("glue", dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset_raw.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    if sample_data:
        train_dataset = dataset["train"].shuffle(seed=42).select(range(train_split))  # keep it small
        eval_dataset = dataset["validation"].select(range(eval_split))
    
    return train_dataset, eval_dataset
