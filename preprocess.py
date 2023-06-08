import spacy
import os
import re
import json
import numpy as np
import config as CFG
from tqdm import tqdm

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

def greedy_selection(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    doc_length = len(doc_sent_list)
    sum_length = len(abstract_sent_list)

    # sum_sents = [_rouge_clean(s).split() for s in abstract_sent_list]
    # doc_sents = [_rouge_clean(s).split() for s in doc_sent_list]
    sum_sents = [[token.lemma_.lower() for token in s] for s in abstract_sent_list]
    doc_sents = [[token.lemma_.lower() for token in s] for s in doc_sent_list]
    sum_sents = [_rouge_clean(" ".join(s)).split() for s in sum_sents]
    doc_sents = [_rouge_clean(" ".join(s)).split() for s in doc_sents]
    # print(sum_sents)
    # print(doc_sents)
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in doc_sents]
    reference_1grams = [_get_word_ngrams(1, [sent]) for sent in sum_sents]
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in doc_sents]
    reference_2grams = [_get_word_ngrams(2, [sent]) for sent in sum_sents]

    coverage = []
    for j in range(sum_length):
        max_rouge = 0.0
        selected = []
        done = False
        for _ in range(doc_length):
            if done: break
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(doc_length):
                if (i in selected):
                    continue
                c = selected + [i]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                candidates_2 = set.union(*map(set, candidates_2))
                rouge_1 = cal_rouge(candidates_1, reference_1grams[j])['r']
                rouge_2 = cal_rouge(candidates_2, reference_2grams[j])['r']
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                done = True
                break
            selected.append(cur_id)
            max_rouge = cur_max_rouge
        coverage.append(selected)
    return coverage

def process_rel_index(doc_sents, sum_sents):
    cov = greedy_selection(doc_sents, sum_sents)
    coverage = np.zeros(len(doc_sents), dtype=int)
    for selected in cov:
        coverage[selected] = 1
    rel_index = np.where(coverage)[0].tolist()
    doc_sents = [x.text for x in doc_sents]
    sum_sents = [x.text for x in sum_sents]
    sample = {"doc_sents": doc_sents, 
              "sum_sents": sum_sents,
              "rel_index": rel_index}
    return sample

if __name__ == '__main__':
    # test("DocNLI/test.jsonl", coref)
    _doc = ['The first vaccine for Ebola was approved by the FDA in 2019 in the US, five years after the initial outbreak in 2014.',
        'To produce the vaccine, scientists had to sequence the DNA of Ebola, then identify possible vaccines, and finally show successful clinical trials.',
        'Scientists say a vaccine for COVID-19 is unlikely to be ready this year, although clinical trials have already started.']
    _sum1 = ['Scientists believe a vaccine for Covid-19 might not be ready this year.', 'The first vaccine for Ebola took 5 years to be approved by the FDA.']
    _sum2 = ['Scientists believe a vaccine for Ebola might not be ready this year.', 'The first vaccine for Ebola took 5 years to be produced by the CBP.']

    data = [{"doc": _doc, "sum": _sum2, "cov": [1, 0, 1]}]

    # nlp=spacy.load("en_core_web_sm", exclude=["tok2vec",'tagger','parser','ner', 'attribute_ruler', 'lemmatizer'])
    # nlp.max_length = 2000000
