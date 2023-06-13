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
import logging
import argparse
# import bitsandbytes as bnb

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder')
parser.add_argument('--data_name', choices=["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"])
parser.add_argument('--lora', default=False, action="store_true")
parser.add_argument('--ptuning', default=False, action="store_true")
parser.add_argument('--freeze', default=False, action="store_true")
parser.add_argument('--resume', default=False, action="store_true")
parser.add_argument('--load_policy', type=LoadPolicy.from_string, choices=list(LoadPolicy))
parser.add_argument('--augment_policy', type=AugmentPolicy.from_string, choices=list(AugmentPolicy))
args = parser.parse_args()
print(args)

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze(-1)
    labels = labels.squeeze(-1)
    predictions = expit(logits) > 0.5
    results = {"ba": balanced_accuracy_score(labels, predictions),
               "f1": f1_score(labels, predictions, average="macro")}
    return results

class CustomTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight]).to(CFG.DEVICE))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    factdata = FactDataset(tokenizer, args.load_policy, args.augment_policy)
    train_set = factdata.load_train(args.data_name)
    val_set = factdata.load_val(args.data_name)

    total_len = len(train_set)
    num_positive = sum([sample["label"][0] for sample in train_set])
    pos_weight = (total_len - num_positive) / num_positive
    print(total_len, num_positive, total_len-num_positive)

    model =  AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME, 
                                                                num_labels=1, 
                                                                problem_type="multi_label_classification",
                                                                ignore_mismatched_sizes=True,)

    if args.lora:
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
    if args.ptuning:
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", 
                                          num_virtual_tokens=20, 
                                          encoder_hidden_size=768)
        model = get_peft_model(model, peft_config)

    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if args.freeze:
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
            param.data = param.data.to(torch.float32)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    # model.to(CFG.DEVICE)
    training_args = TrainingArguments(output_dir=args.output_folder, 
                                      evaluation_strategy="steps",
                                      save_strategy="steps",
                                      eval_steps=500,
                                      save_steps=500,
                                      save_total_limit=2,
                                      metric_for_best_model='ba',
                                      load_best_model_at_end=True,
                                      num_train_epochs=CFG.EPOCH,
                                      optim="paged_adamw_8bit",
                                      learning_rate=CFG.LR,
                                    #   lr_scheduler_type='constant',
                                      per_device_train_batch_size=CFG.BATCH_SIZE,
                                      tf32=True,)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # benchmarking
    for name in factdata.data_names:
        print(name)
        test_set = factdata.load_test(name)
        print(trainer.evaluate(test_set))
