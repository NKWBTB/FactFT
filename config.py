import torch

MODEL_NAME = 'henry931007/mfma'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 5
EPOCH = 10
LR = 1e-3
EXP_PATH = "exp/"