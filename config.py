import torch

MODEL_NAME = 'henry931007/mfma'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 32
EPOCH = 10
LR = 2e-5
EXP_PATH = "exp/"