import os
import json
from sklearn.model_selection import StratifiedGroupKFold
from utils import load_jsonl
from enum import Enum
from datasets import Dataset

class LoadPolicy(Enum):
    Train_1_Val_n = 1
    Train_n_Val_1 = 2
    Train_1_Val_1 = 3
    k_5_Fold = 4

    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return LoadPolicy[s]
        except KeyError:
            raise ValueError()

class AugmentPolicy(Enum):
    NoAugment = 1
    NegAugmentOnly = 2
    AllAugment = 3

    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return AugmentPolicy[s]
        except KeyError:
            raise ValueError()

class FactDataset:
    def __init__(self, tokenizer, load_policy, aug_policy, use_original=True):
        self.data_folder = "data/merge/"
        self.data_names = set(["cogensumm", "xsumfaith", "polytope", "factcc", "summeval", "frank"])
        self.tokenizer = tokenizer
        self.load_policy = load_policy
        self.aug_policy = aug_policy
        self.use_original = use_original
        self.remove_columns = ["augment", "original_doc"]

        if load_policy == LoadPolicy.k_5_Fold:
            groups = []
            self.data = self.load_data(sorted(self.data_names), "val", groups=groups)
            labels = [sample["label"] for sample in self.data]
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            self.train_idx = []
            self.test_idx = []
            for train_idx, test_idx in skf.split(X=[0]*len(labels), y=labels, groups=groups):
                self.train_idx.append(train_idx)
                self.test_idx.append(test_idx)

    def preprocess_function(self, examples):
        if self.use_original:
            examples["document"] = examples["original_doc"]
        tokenized_data = self.tokenizer(examples["document"], 
                        examples["claim"], 
                        padding="max_length", 
                        truncation="longest_first")
        labels = [[label] for label in examples["label"]]
        tokenized_data["label"] = labels
        return tokenized_data
    
    def load_data(self, desired_set, cut, groups=None):
        print(desired_set, cut)
        datas = []
        for name in desired_set:
            data = load_jsonl(os.path.join(self.data_folder, f"{name}_{cut}.jsonl"))
            if self.aug_policy == AugmentPolicy.NoAugment:
                data = [sample for sample in data if not sample["augment"]]
            elif self.aug_policy == AugmentPolicy.NegAugmentOnly:
                data = [sample for sample in data if (not sample["augment"]) or \
                                                     (sample["augment"] and sample["label"]==0)]
            datas.extend(data)
        
        if groups is not None:
            doc_set = set([sample["original_doc"] for sample in datas])
            doc_dict = dict(zip(sorted(doc_set), range(len(doc_set))))
            for sample in datas:
                groups.append(doc_dict[sample["original_doc"]])
                
        datas = Dataset.from_list(datas)
        datas = datas.map(self.preprocess_function, 
                          remove_columns=self.remove_columns,
                          batched=True)
        return datas

    def load_train(self, name=None, k=None):
        if self.load_policy == LoadPolicy.k_5_Fold:
            print("KFold train", k)
            return self.data.select(self.train_idx[k])
        
        desired_set = None
        if self.load_policy == LoadPolicy.Train_n_Val_1:
            desired_set = self.data_names - set([name])
        elif self.load_policy == LoadPolicy.Train_1_Val_n or \
            self.load_policy == LoadPolicy.Train_1_Val_1:
            desired_set = set([name])
        self.train_set = self.load_data(desired_set, 'val')
        return self.train_set
    
    def load_val(self, name=None, k=None):
        if self.load_policy == LoadPolicy.k_5_Fold:
            print("KFold test", k)
            return self.data.select(self.test_idx[k])

        desired_set = None
        if self.load_policy == LoadPolicy.Train_n_Val_1:
            desired_set = set([name])
        elif self.load_policy == LoadPolicy.Train_1_Val_n:
            desired_set = self.data_names - set([name])
        elif self.load_policy == LoadPolicy.Train_1_Val_1:
            desired_set = set([name])
            self.val_set = self.load_data(desired_set, 'test')
            return self.val_set
        self.val_set = self.load_data(desired_set, 'val')
        return self.val_set
    
    def load_test(self, name):
        desired_set = set([name])
        test_set = self.load_data(desired_set, 'test')
        return test_set
    
if __name__ == '__main__':
    from transformers import AutoTokenizer
    MODEL_NAME = 'microsoft/deberta-v2-xlarge-mnli'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    factdata = FactDataset(tokenizer, LoadPolicy.Train_n_Val_1, AugmentPolicy.NoAugment)
    train = factdata.load_train('factcc')
    val = factdata.load_val('factcc')
