import json
from tqdm import tqdm
import os.path
from difflib import SequenceMatcher
from utils import load_jsonl

def sequence_match(text_1, text2):
    s = SequenceMatcher(None, text_1, text2)
    if s.real_quick_ratio() > 0.95:
        if s.quick_ratio() > 0.95:
            if s.ratio() > 0.95:
                return True
    return False

def check_overlap(test_documents, val_file, use_joblib=False):
    """
    This function check if the test file and validation file share the same document
    :param test_documents: list of test documents
    :param val_file: Path of validation file
    :return overlap_index: The duplicate index in validation file
    :return count: number of overlap pairs
    """
    if use_joblib:
        from joblib import Parallel, delayed
    
    val_list = load_jsonl(val_file)
    print(f"val file has {len(val_list)} elements")

    count = 0
    overlap_index = []
    for i in tqdm(range(len(val_list))):
        doc_val = val_list[i]['document']
        # Sequential
        if not use_joblib:
            for doc_test in test_documents:
                if sequence_match(doc_val, doc_test):
                    count += 1
                    overlap_index.append(i)
                    break
        else:
            result = Parallel(n_jobs=15)(delayed(sequence_match)(doc_val, doc_test) for doc_test in test_documents)
            if any(result):
                count += 1
                overlap_index.append(i)
    return overlap_index, count


def delete_overlap_item(val_file, val_file_out, index2del):
    """
    This function delete the duplicate items from validation file based on the given index
    and generate the new validation file without duplicate
    :param val_file: The path of validation file
    :param val_file_out: The path of validation file after delete duplicates
    :param index2del: The duplicate's index
    """
    with open(val_file, 'r', encoding="utf-8") as f:
        val_list = f.readlines()

    with open(val_file_out, 'w', encoding="utf-8") as f:
        outputs = [val_list[i] for i in range(len(val_list)) if i not in index2del]
        f.write(''.join(val_list))

    print(f"val file has {len(val_list)} items, after delete overlap it has {len(outputs)} items")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data_names = ['cogensumm', 'factcc', 'frank', 'polytope', 'summeval', 'xsumfaith']
    DATA_FOLDER = 'data/'
    RAW_FOLDER = 'raw/'
    AUGMENT_FOLDER = 'augment/'
    DUMP_FOLDER = 'augment_filtered/'
    test_documents = set()
    for name in data_names:
        test_file = os.path.join(DATA_FOLDER, RAW_FOLDER, name + '_test.jsonl')
        test_set = load_jsonl(test_file)
        print("original test size", len(test_set))
        test_docs = set([sample["document"] for sample in test_set])
        test_documents.update(test_docs)
        print("unique test size", len(test_documents), len(test_docs))
    
    paths_val_aug = []
    paths_val_dump = []

    for name in data_names:
        path_val_aug = os.path.join(DATA_FOLDER, AUGMENT_FOLDER, name + '_val.jsonl')
        path_val_dump = os.path.join(DATA_FOLDER, DUMP_FOLDER, name + '_val.jsonl')
        paths_val_aug.append(path_val_aug)
        paths_val_dump.append(path_val_dump)

    # overlap_index, count = check_overlap(paths_test[0], paths_val[1])
    # print("Overlap indexes are")
    # print(overlap_index)
    # print("Number of overlap is")
    # print(count)
    overlap_index_all = []
    for i, path_val in enumerate(paths_val_aug):
        print(path_val)
        overlap_index = []
        overlap_index, count = check_overlap(test_documents, path_val, use_joblib=True)
        overlap_index_all.append(overlap_index)
        print(count, overlap_index)

    for input_path, output_path, index2del in zip(paths_val_aug, paths_val_dump, overlap_index_all):
        delete_overlap_item(input_path, output_path, index2del)
