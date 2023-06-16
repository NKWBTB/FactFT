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
from peft import get_peft_model, LoraConfig, PromptEncoderConfig
from fact_dataset import FactDataset, LoadPolicy, AugmentPolicy

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
LORA = True
PTUNING = False

if __name__ == '__main__':
    factdata = FactDataset(tokenizer, load_policy=None, aug_policy=None, use_original=False)
    model =  AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME, 
                                                                num_labels=1, 
                                                                problem_type="multi_label_classification",
                                                                ignore_mismatched_sizes=True,)

    if LORA:
        peft_config = LoraConfig(
            task_type = "SEQ_CLS",
            r = 8,
            lora_alpha = 8,
            lora_dropout = 0.1,
            inference_mode = False, 
            target_modules='.*attention\.self\.(query_proj|value_proj)',
            bias='lora_only'
        )
        model = get_peft_model(model, peft_config)
    if PTUNING:
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", 
                                          num_virtual_tokens=20, 
                                          encoder_hidden_size=768)
        model = get_peft_model(model, peft_config)
    
    CHECKPOINT = 'exp/5e5/NoAugment/4/'
    model.load_adapter(CHECKPOINT, adapter_name='default', subfolder='checkpoint-800')
    trainer = Trainer(
        model = model,
        args = TrainingArguments(
            per_device_eval_batch_size=32,
            output_dir='./test_trainer'
        ))
    
    DUMP = 'pred.json'
    results = {}
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score
    from scipy.special import expit

    for name in factdata.data_names:
        print(name)
        test_set = factdata.load_test(name)
        preds, labels, metrics = trainer.predict(test_set)
        preds = np.array(expit(preds) > 0.5, dtype=int).squeeze(-1).tolist()
        labels = labels.squeeze(-1).tolist()
        cmat = confusion_matrix(labels, preds)
        results[name] = {"preds": preds, "labels": labels, "cmat": cmat.tolist()}
        print(confusion_matrix(labels, preds))
