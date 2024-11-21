from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from tools.data import EnsembleDataset, get_predictions_df
from torch.utils.data import DataLoader

SEED = 4376823

torch.manual_seed(SEED)
np.random.seed(SEED)
# torch.set_default_device(torch.device("cuda"))

non_feat_cols = ["model", "text", "id", "label"]

generators = ["chatGPT", "cohere", "davinci", "dolly", "bloomz"]
strategies = ["tfidf_gs", "lang_gs", "bert"]

# `train` (all of train set) or `custom-dev` (unseen data from train for all sub-classifiers)
# if training on custom-dev, custom CV in grid search should be disabled
df_train = get_predictions_df(
    "custom-train", strategies=strategies, generators=generators
)
df_dev = get_predictions_df(
    "custom-dev", strategies=strategies, generators=generators
).reindex(columns=df_train.columns)
df_test = get_predictions_df(
    "test", strategies=strategies, generators=generators
).reindex(columns=df_train.columns)

ds_train = EnsembleDataset(df_train.drop(columns=non_feat_cols), df_train[["label"]])
ds_dev = EnsembleDataset(df_dev.drop(columns=non_feat_cols), df_dev[["label"]])
ds_test = EnsembleDataset(df_test.drop(columns=non_feat_cols), df_test[["label"]])

train_loader = DataLoader(ds_train, batch_size=128, shuffle=True)
dev_loader = DataLoader(ds_dev, batch_size=16, shuffle=False)
test_loader = DataLoader(ds_test, batch_size=16, shuffle=False)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.sm = nn.LogSoftmax(dim=-1)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, X):
        y = self.layers(X)
        y = self.sm(y)
        return y


input_size = len(df_train.drop(columns=non_feat_cols).columns)
model = Model(input_size=input_size, hidden_size=64)
criterion = nn.NLLLoss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)


def ev(loader=dev_loader, fname=""):
    model.eval()
    pred = []
    target = []
    with torch.no_grad():
        for batch in loader:
            feats, labels = batch
            out = model(feats)
            out = out.argmax(dim=-1)
            target.extend(labels.cpu())
            pred.extend(out.cpu())
    report = classification_report(target, pred, digits=5)
    return report


n_epochs = 5

for i in range(n_epochs):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        feats, labels = batch
        out = model(feats)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    report_dev = ev(loader=dev_loader, fname=f"e{i+1}_dev")
    report_test = ev(loader=test_loader, fname=f"e{i+1}_test")

    with open(f"reports/ensemble-nn{i}.txt", "w") as f:
        f.write(f"DEV\n\n{report_dev}\n\n\n\nTEST\n\n{report_test}")
