import numpy as np
import json

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n') 

def shift_np(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result