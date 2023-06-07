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
from peft import get_peft_model, LoraConfig, PromptEncoderConfig
import logging
import bitsandbytes as bnb
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

MODEL_NAME = 'microsoft/deberta-v2-xlarge-mnli'
MODEL_NAME = './RedPajama-INCITE-Base-3B-v1'
DATA_TRAIN = 'factcc'
REMOVE_COLUMNS = ['filepath', 'id', 'annotations', 'dataset', 'origin']
FREEZE = False
LORA = True
PTUNING = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([15/85]))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
                                                                ignore_mismatched_sizes=True,
                                                                load_in_8bit=True, 
                                                                device_map='auto',)
    # torch.save(model.state_dict(), "pytorch_model.bin")
    # import pdb
    # pdb.set_trace()

    if LORA:
        peft_config = LoraConfig(
            task_type = "SEQ_CLS",
            r = 8,
            lora_alpha = 8,
            lora_dropout = 0.1,
            inference_mode = False, 
            # target_modules='.*attention\.self\.(query_proj|value_proj)',
            bias='lora_only'
        )
        model = get_peft_model(model, peft_config)
    if PTUNING:
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", 
                                          num_virtual_tokens=20, 
                                          encoder_hidden_size=768)
        model = get_peft_model(model, peft_config)

    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if FREEZE:
            if not name.startswith("deberta.encoder.layer.23.attention.self") \
                and not name.startswith("classifier"):
                param.requires_grad = False
        print(name, param.requires_grad)
        num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    # model.to(CFG.DEVICE)
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      evaluation_strategy="epoch",
                                      num_train_epochs=CFG.EPOCH,
                                      learning_rate=CFG.LR,
                                      lr_scheduler_type='constant',
                                      per_device_train_batch_size=CFG.BATCH_SIZE,
                                      fp16=True,)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()
