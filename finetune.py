import argparse
import os
import subprocess
import sys
import torch
import numpy as np

import setup
setup.setup_environment()
import dataset

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from adapters import AutoAdapterModel, AdapterTrainer


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "sst2"
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

#@title Define Trainer helper

def _compute_metrics(p: EvalPrediction):
    """Function to compute Accuracy and F1 score."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def _train_and_eval(model, run_name, learning_rate, epochs, weight_decay, train_split, eval_split, tokenizer, adapter_trainer=False):

    """Get the trainer to perform finetuning."""

    args = TrainingArguments(
      output_dir=f"./results/{run_name}",
      eval_strategy="steps",
      eval_steps=50,
      save_strategy="epoch",
      learning_rate=learning_rate,
      # warmup_ratio=0.1,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=epochs,
      weight_decay=weight_decay,
      logging_dir=f"./logs/{run_name}",
      logging_steps=50,
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
    if adapter_trainer:
        trainer = AdapterTrainer(
            model=model,
            args=args,
            train_dataset=train_split,
            eval_dataset=eval_split,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics
        )
    return trainer

def fine_tune(model_name, learning_rate, epochs, weight_decay, train_split, eval_split, freeze_encoder):
    """This function is used to finetune the given model using traditional SFT.
    Args:
        model_name: model to be used.
        learning_rate: learning rate to be used in training.
        epochs: Number of epochs to be used.
        weight_decay: weight decay to used in training.
        train_split: train the model on this data.
        eval_split: evaluate the model on this data.
        freeze_encoder: If true will freeze the encoder weights for training.
    """
    print('Starting Fine tuning.')
    model_full = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if freeze_encoder:
        for layer in model_full.distilbert.transformer.layer:
            for param in layer.parameters():
                param.requires_grad = False

    print("Full FT: Trainable params:", sum(p.numel() for p in model_full.parameters() if p.requires_grad))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = _train_and_eval(model_full, "full_ft", learning_rate, epochs, weight_decay, train_split, eval_split, tokenizer)
    trainer.train()


def fine_tune_with_adapters(
    model_name,
    learning_rate,
    epochs,
    weight_decay,
    train_split,
    eval_split,
    number_of_adapters,
    reduction_factor,
    activation_function
):
    """This function is used to fine tune a model using adapters.
    Args:
        model_name: model to be used.
        learning_rate: learning rate to be used in training.
        epochs: Number of epochs to be used.
        weight_decay: weight decay to used in training.
        train_split: train the model on this data.
        eval_split: evaluate the model on this data.
        reduction_factor: Reduce the hidden dim size by this factor. Ref: https://docs.adapterhub.ml/methods.html#bottleneck-adapters.
        activation_function: activation function to be used in adapters.
    """
    print('Starting Fine tuning using Adapters')
    model_with_adapters = AutoAdapterModel.from_pretrained(model_name)
    for i in range(number_of_adapters):
        model_with_adapters.add_adapter(f"adapter_{i}")
        model_with_adapters.train_adapter(f"adapter_{i}")
    model_with_adapters.add_classification_head("Custom_head", num_labels=2)
    total_trainable_params = sum(p.numel() for p in model_with_adapters.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = _train_and_eval(model_with_adapters, "adapter_ft", learning_rate, epochs, weight_decay, train_split, eval_split, tokenizer, adapter_trainer=True)
    trainer.train()


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model to use", default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", type=str, help="Dataset to use", default="sst2")
    parser.add_argument("--learning_rate", type=float, help="learning rate to use in finetuning", default=2e-5)
    parser.add_argument("--epochs", type=int, help="Number of epochs to use", default=3)
    parser.add_argument("--weight_decay", type=float, help="Regularize the weights", default=0.01)
    parser.add_argument("--sample_data", action="store_true", help="Whether to sample data for training and eval. If True will sample 500 samples for training and 200 samples for eval")
    # SFT based arguments
    parser.add_argument("--perform_sft", action="store_true", help="Perform traditional Fine Tuning.")
    parser.add_argument("--freeze_encoder", action="store_true", help="Whether to freeze the encoder")
    # Adapters based arguments
    parser.add_argument("--use_adapters", action="store_true", help="Whether to perform finetuning using adapters.")
    parser.add_argument("--number_of_adapters", type=int, default=1, help="Number of adapters per transformer layer.")
    parser.add_argument("--reduction_factor", type=int, default=16, help="Reduce the hidden diemnsions by this factor. For more details:https://docs.adapterhub.ml/methods.html#bottleneck-adapters")
    parser.add_argument("--activation_function", type=str, default="relu", help="Type of activation functions to use in adapter layer.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = define_args()
    # Train split: 500
    # eval split: 200
    train_split, eval_split = dataset.get_train_val_split(model_name=args.model_name, dataset_name=args.dataset_name, sample_data=args.sample_data)
    print("Train Data:", train_split.shape)
    print("Eval Data:", eval_split.shape)
    if args.perform_sft:
        fine_tune(args.model_name, args.learning_rate, args.epochs, args.weight_decay, train_split, eval_split, args.freeze_encoder)
    if args.use_adapters:
        fine_tune_with_adapters(args.model_name, args.learning_rate, args.epochs, args.weight_decay, train_split, eval_split, args.number_of_adapters, args.reduction_factor, args.activation_function)
