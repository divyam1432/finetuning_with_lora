import argparse
import os
import subprocess
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

import setup
import dataset


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "sst2"
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

#@title Define Trainer helper

def _compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def _train_and_eval(model, run_name, learning_rate, epochs, weight_decay, train_split, eval_split, tokenizer):

    args = TrainingArguments(
      output_dir=f"./results/{run_name}",
      eval_strategy="epoch",
      save_strategy="epoch",
      learning_rate=learning_rate,
      # warmup_ratio=0.1,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=epochs,
      weight_decay=weight_decay,
      logging_dir=f"./logs/{run_name}",
      logging_steps=10,
      report_to=["tensorboard"],
      # fp16=torch.cuda.is_available(),
      disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics
    )
    return trainer

def fine_tune(model_name, learning_rate, epochs, weight_decay, train_split, eval_split):
    print('Starting Fine tuning.')
    model_full = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    for layer in model_full.distilbert.transformer.layer:
        for param in layer.parameters():
            param.requires_grad = False

    print("Full FT: Trainable params:", sum(p.numel() for p in model_full.parameters() if p.requires_grad))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = _train_and_eval(model_full, "full_ft", learning_rate, epochs, weight_decay, train_split, eval_split, tokenizer)
    trainer.train()

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model to use", default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, help="Dataset to use", default="sst2")
    parser.add_argument("--learning_rate", type=float, help="learning rate to use in finetuning", default=2e-5)
    parser.add_argument("--epochs", type=int, help="Number of epochs to use", default=3)
    parser.add_argument("--weight_decay", type=float, help="Regularize the weights", default=0.01)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = define_args()
    setup.setup_environment()
    # Train split: 500
    # eval split: 200
    train_split, eval_split = dataset.get_train_val_split(model_name=args.model_name, dataset_name=args.dataset_name)
    print("Train Data:", train_split.shape)
    print("Eval Data:", eval_split.shape)
    fine_tune(args.model_name, args.learning_rate, args.epochs, args.weight_decay, train_split, eval_split)
