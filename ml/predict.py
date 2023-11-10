import torch

from python.ml.logistic_regression import LogisticRegression

PATH = "/tmp/model-6656168128204039283"
model = LogisticRegression(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

