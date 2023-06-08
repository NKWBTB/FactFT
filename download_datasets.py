from summac.benchmark import SummaCBenchmark
from utils import dump2jsonl
from tqdm import tqdm
import preprocess as pre
import spacy
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
    for cut in ["val", "test"]:
        benchmark = SummaCBenchmark(benchmark_folder="./summac_benchmark/", cut=cut)
        for dataset in benchmark.datasets:
            name = dataset["name"]
            print(name, cut)
            processed_dataset = [process_single_sample(sample, nlp) for sample in tqdm(dataset["dataset"])]
            dump2jsonl(processed_dataset, os.path.join("data/raw/", "_".join([name, cut])+'.jsonl'))

