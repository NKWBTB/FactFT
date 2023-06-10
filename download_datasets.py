from summac.benchmark import SummaCBenchmark
from utils import dump2jsonl, load_jsonl
from tqdm import tqdm
import preprocess as pre
import spacy
import augmentation as aug
import os

def process_single_sample(sample, nlp):
    _doc = sample["document"]
    _sum = sample["claim"]

    doc_nlp = nlp(_doc)
    doc_sents = list(doc_nlp.sents)
    sum_nlp = nlp(_sum)
    sum_sents = list(sum_nlp.sents)

    processed_idx = pre.process_rel_index(doc_sents, sum_sents)
    updated_sample = sample | processed_idx
    return updated_sample

if __name__ == '__main__':
    spacy.require_gpu()
    nlp=spacy.load("en_core_web_trf")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2000000

    dataset_names = set()
    DATA_FOLDER = 'data/'
    # Process the raw dataset
    RAW_FOLDER = "raw/"
    for cut in ["val", "test"]:
        benchmark = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut=cut)
        for dataset in benchmark.datasets:
            name = dataset["name"]
            dataset_names.add(name)
            print(name, cut)
            dump_to = os.path.join(DATA_FOLDER, RAW_FOLDER, "_".join([name, cut])+'.jsonl')
            if os.path.exists(dump_to): 
                print("skip")
                continue
            processed_dataset = [process_single_sample(sample, nlp) for sample in tqdm(dataset["dataset"])]
            dump2jsonl(processed_dataset, dump_to)

    # Augment the validation set
    # import tiktoken
    # encoding = tiktoken.get_encoding('p50k_base')
    # encoding = tiktoken.encoding_for_model("text-babbage-001")
    # max_token_len = 0
    AUG_FOLDER = "augment/"
    cut = 'val'
    benchmark = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut=cut)
    for name in dataset_names:
        print(name, cut)
        dump_to = os.path.join(DATA_FOLDER, AUG_FOLDER, "_".join([name, cut])+'.jsonl')
        if os.path.exists(dump_to): 
            print("skip")
            continue
        input_path = os.path.join(DATA_FOLDER, RAW_FOLDER, "_".join([name, cut])+'.jsonl')
        data = load_jsonl(input_path)
        for sample in tqdm(data):
            doc_sents = sample["doc_sents"]
            sum_sents = sample["sum_sents"]
            rel_index = sample["rel_index"]
            rel_len = len(rel_index)
            sents2aug = [doc_sents[idx] for idx in rel_index] + sum_sents
            aug_sents = aug.rephrase(sents2aug)
            # Augment relevant sentences in document by rephrasing
            doc_augment_dict = {rel_index[i] : aug_sents[i] for i in range(rel_len)}
            # for idx in rel_index:
            #     sent2aug = doc_sents[idx]
            #     max_token_len = max([max_token_len, len(encoding.encode(sent2aug))])
            #     doc_augment_dict[idx] = aug.rephrase(input_sent=sent2aug)
            
            # Augment summary sentences
            sum_len = len(sum_sents)
            sum_augment_dict = {idx : aug_sents[idx+rel_len] for idx in range(sum_len)}
            # for idx in range(sum_len):
            #     sent2aug = sum_sents[idx]
            #     max_token_len = max([max_token_len, len(encoding.encode(sent2aug))])
            #     sum_augment_dict[idx] = aug.rephrase(input_sent=sent2aug)

            sample["doc_augments"] = doc_augment_dict
            sample["sum_augments"] = sum_augment_dict
        dump2jsonl(data, dump_to)
    # print("max_token_len", max_token_len)
