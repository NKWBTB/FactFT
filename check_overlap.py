import json
from tqdm import tqdm
import os.path
from difflib import SequenceMatcher


def check_overlap(test_file, val_file):
    """
    This function check if the test file and validation file share the same document
    :param test_file: Path of test file
    :param val_file: Path of validation file
    :return overlap_index: The duplicate index in validation file
    :return count: number of overlap pairs
    """
    with open(test_file, 'r', encoding="utf-8") as json_file_test:
        json_list_test = list(json_file_test)

    with open(val_file, 'r', encoding="utf-8") as json_file_val:
        json_list_val = list(json_file_val)

    print(f"test file has {len(json_list_test)} elements")
    print(f"val file has {len(json_list_val)} elements")

    count = 0
    count_val = 0
    overlap_index = []
    for json_str_val in tqdm(json_list_val):
        val_content = json.loads(json_str_val)
        doc_val = val_content['document']
        for json_str_test in json_list_test:
            test_content = json.loads(json_str_test)
            doc_test = test_content['document']
            s = SequenceMatcher(None, doc_val, doc_test)
            if s.real_quick_ratio() > 0.95:
                if s.quick_ratio() > 0.95:
                    if s.ratio() > 0.95:
                        # print(s.quick_ratio())
                        # print("Val file is")
                        # print(doc_val)
                        # print("Test file is ")
                        # print(doc_test)
                        count += 1
                        overlap_index.append(count_val)
                        break
        count_val += 1
    return overlap_index, count


def delete_overlap_item(val_file, val_file_out, index):
    """
    This function delete the duplicate items from validation file based on the given index
    and generate the new validation file without duplicate
    :param val_file: The path of validation file
    :param val_file_out: The path of validation file after delete duplicates
    :param index: The duplicate's index
    """
    with open(val_file, 'r', encoding="utf-8") as json_file_val:
        json_list_val = list(json_file_val)

    with open(val_file_out, 'w', encoding="utf-8") as json_file_val_out:
        count = 0
        for i in range(0, len(json_list_val)):
            if i not in index:
                count += 1
                json_file_val_out.write(json.dumps(json_list_val[i]) + "\n")

    print(f"val file has {len(json_list_val)} items, after delete overlap it has {count} items")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    filenames = ['cogensumm', 'factcc', 'frank', 'polytope', 'summeval', 'xsumfaith']

    paths_val = []
    paths_test = []
    paths_val_aug = []
    paths_val_norep = []

    for filename in filenames:
        path_val = os.path.join('raw\\raw', filename + '_val.jsonl')
        path_test = os.path.join('raw\\raw', filename + '_test.jsonl')
        path_val_aug = os.path.join('augment', filename + '_val.jsonl')
        path_val_norep = os.path.join('augment', filename + '_val_norep.jsonl')
        paths_val.append(path_val)
        paths_test.append(path_test)
        paths_val_aug.append(path_val_aug)
        paths_val_norep.append(path_val_norep)

    # overlap_index, count = check_overlap(paths_test[0], paths_val[1])
    # print("Overlap indexes are")
    # print(overlap_index)
    # print("Number of overlap is")
    # print(count)
    overlap_index_all = []
    for i, path_val in enumerate(paths_val):
        overlap_index_total = []
        for path_test in paths_test:
            overlap_index, count = check_overlap(path_test, path_val)
            overlap_index_total = overlap_index_total + overlap_index
            # if count > 0:
            #     print(f"{count} pairs has similarity over 0.95")
            #     print("(" + path_test + ") (" + path_val + ") has overlap")
        overlap_index_sorted = [*set(overlap_index_total)]
        print(overlap_index_sorted)
        overlap_index_all.append(overlap_index_sorted)

    for i, path in enumerate(paths_val_aug):
        delete_overlap_item(paths_val_aug[i], paths_val_norep[i], overlap_index_all[i])
