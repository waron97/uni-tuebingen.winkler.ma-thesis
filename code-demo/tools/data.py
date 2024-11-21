import json
import os
import torch
from torch.utils.data import Dataset


import pandas as pd
from .paths import (
    data_dev,
    data_test,
    data_train,
    feats_dev,
    feats_test,
    feats_train,
    feats_joined,
    custom_train,
    custom_dev,
)


def get_df(split):
    p1 = ""
    labels, texts, models = {}, {}, {}
    if split == "train":
        p1 = data_train
    elif split == "dev":
        p1 = data_dev
    elif split == "test":
        p1 = data_test
    elif split == "custom-train":
        p1 = custom_train
    elif split == "custom-dev":
        p1 = custom_dev

    with open(p1, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                labels[j["id"]] = j["label"]
                texts[j["id"]] = j.get("text", None)
                models[j["id"]] = j.get("model", None)
            except Exception as e:
                pass

    d = []

    for _id in labels:
        r = {"label": labels[_id], "text": texts[_id], "model": models[_id], "id": _id}

        d.append(r)

    df = pd.DataFrame(d)
    return df


def get_feats_df(split):
    p = ""
    if split == "train":
        p = feats_train
    elif split == "dev":
        p = feats_dev
    elif split == "test":
        p = feats_test
    elif split == "train+dev":
        p = feats_joined

    d = []

    with open(p, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                d.append(j)
            except:
                pass

    df = pd.DataFrame(d)
    df = df.drop(columns=["split", "t_kup", "a_kup_pw", "a_kup_ps"])
    return df


def get_predictions_df(
    split,
    generators=["chatGPT", "cohere", "davinci", "dolly", "bloomz"],
    strategies=["tfidf_gs", "bert", "lang_gs"],
):
    base: pd.DataFrame = get_df(split)
    for g in generators:
        for s in strategies:
            id_ = f"{s}-{g}"
            try:
                p = os.path.join("predictions", f"{id_}.jsonl")
                df = pd.read_json(p, lines=True)
                split_name = {
                    "test": "test",
                    "train": "train",
                    "dev": "dev",
                    "custom-train": "train",
                    "custom-dev": "dev",
                }[split]
                df = df[df.split == split_name]
                df = df[["id", "pred"]].rename(columns={"pred": id_})
                base = base.merge(df, how="left", on="id", validate="1:1")
            except Exception as err:
                print(id_, err)
    return base


class EnsembleDataset(Dataset):
    def __init__(self, input_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
        super().__init__()
        self.input_df = input_df
        self.labels_df = labels_df
        self.non_feat_cols = ["model", "text", "id", "label"]

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, index):
        feats: pd.Series = self.input_df.iloc[index]
        label = self.labels_df.iloc[index]["label"]
        return (
            torch.tensor(feats, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )
