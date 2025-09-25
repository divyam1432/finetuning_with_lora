import os
import subprocess
import sys

import setup
import dataset

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "sst2"

if __name__ == "__main__":
    setup.setup_environment()
    # Train split: 500
    # eval split: 200
    train_split, eval_split = dataset.get_train_val_split(model_name=MODEL_NAME, dataset_name=DATASET_NAME)
    print("Train Data:", train_split.shape)
    print("Eval Data:", eval_split.shape)
