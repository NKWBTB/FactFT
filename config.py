import torch

MODEL_NAME = 'microsoft/deberta-v2-xlarge-mnli'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 5
EPOCH = 10
LR = 1e-5