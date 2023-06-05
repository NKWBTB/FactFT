import config as CFG
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
from scipy.special import expit
from datasets import Dataset
from summac.benchmark import SummaCBenchmark
import logging

MODEL_NAME = 'microsoft/deberta-large-mnli'
DATA_TRAIN = 'factcc'
REMOVE_COLUMNS = ['filepath', 'id', 'annotations', 'dataset', 'origin']
FREEZE = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    tokenized_data = tokenizer(examples["document"], 
                     examples["claim"], 
                     padding="max_length", 
                     truncation="longest_first")
    labels = [[label] for label in examples["label"]]
    tokenized_data["label"] = labels
    return tokenized_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze(-1)
    labels = labels.squeeze(-1)
    predictions = expit(logits) > 0.5
    results = {"ba": balanced_accuracy_score(labels, predictions),
               "f1": f1_score(labels, predictions, average="macro")}
    return results

if __name__ == '__main__':
    benchmark_val = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut="val")
    benchmark_test = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut="test")
    factcc_val = benchmark_val.get_dataset('factcc')
    factcc_test = benchmark_test.get_dataset('factcc')
    factcc_val = Dataset.from_list(factcc_val)
    factcc_test = Dataset.from_list(factcc_test)
    train_set = factcc_val.map(preprocess_function, 
                               remove_columns=REMOVE_COLUMNS,
                               batched=True)
    test_set = factcc_test.map(preprocess_function,
                               remove_columns=REMOVE_COLUMNS, 
                               batched=True)
    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, 
                                                                num_labels=1, 
                                                                problem_type="multi_label_classification",
                                                                ignore_mismatched_sizes=True)
    
    if FREEZE:
        for name, param in model.named_parameters():
            if not name.startswith("deberta.encoder.layer.23.attention.self"):
                # and not name.startswith("encoder.layer.11.intermediate"):
                param.requires_grad = False
            print(name, param.requires_grad)

    model.to(CFG.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG.LR)
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      evaluation_strategy="epoch",
                                      num_train_epochs=CFG.EPOCH,
                                      per_device_train_batch_size=CFG.BATCH_SIZE)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
        optimizers = (optimizer, None)
    )

    trainer.train()
