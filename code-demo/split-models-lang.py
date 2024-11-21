import json
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report

SEED = 4376823
torch.manual_seed(SEED)
np.random.seed(SEED)

from tools.data import get_df, get_feats_df

models = ["chatGPT", "cohere", "davinci", "dolly", "bloomz"]

feats_train = get_feats_df("train+dev")
feats_test = get_feats_df("test")

df_train = get_df("custom-train").merge(
    feats_train, how="left", on="id", validate="one_to_one"
)
df_dev = (
    get_df("custom-dev")
    .merge(feats_train, how="left", on="id", validate="one_to_one")
    .reindex(df_train.columns, axis="columns")
)
df_test = (
    get_df("test")
    .merge(feats_test, how="left", on="id", validate="one_to_one")
    .reindex(df_train.columns, axis="columns")
)

non_feat_cols = ["model", "text", "id", "label"]

search_parameters = {
    "n_estimators": [50, 100, 500, 1000, 1500, 2000, 2500, 3000],
    "max_depth": [None],
    "max_features": ["sqrt"] + [4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [5, 8, 10, 15, 20],
    "random_state": [SEED],
}


def ev_model(
    model: RandomForestClassifier,
    dataset: pd.DataFrame,
    source: str,
    split_id: str,
):
    index_h = 0 if model.classes_[0] == 0 else 1
    index_m = 0 if index_h == 1 else 1

    X = dataset.drop(columns=non_feat_cols)
    Y_true = dataset["label"]
    Y_pred_probs = model.predict_proba(X)
    scores = Y_pred_probs[:, index_m]
    Y_pred = np.argmax(Y_pred_probs, axis=1)

    rows = []
    for i, score in enumerate(scores):
        item = dataset.iloc[i].to_dict()
        rows.append({"split": split_id, "id": item["id"], "pred": float(score)})

    return classification_report(Y_true, Y_pred), rows


for source in models:
    pos = df_train[df_train.model == source]
    neg = df_train[df_train.model == "human"].sample(pos.shape[0], random_state=SEED)

    X_train = pd.concat([pos, neg]).drop(columns=non_feat_cols)
    Y_train = pd.concat([pos, neg])["label"]

    dev_pos = df_dev[df_dev.model == source]
    dev_neg = df_dev[df_dev.model == "human"][: dev_pos.shape[0]]

    X_dev = pd.concat([dev_pos, dev_neg]).drop(columns=non_feat_cols)
    Y_dev = pd.concat([dev_pos, dev_neg])["label"]

    X = pd.concat([X_train, X_dev])
    Y = pd.concat([Y_train, Y_dev])

    estimator = RandomForestClassifier()
    model = GridSearchCV(
        estimator=estimator,
        param_grid=search_parameters,
        n_jobs=-1,
        scoring="accuracy",
        cv=[
            [
                list(range(0, X_train.shape[0])),
                list(range(X_train.shape[0], X.shape[0])),
            ]
        ],
        verbose=2,
    )
    model.fit(X, Y)

    report_dev, scores_dev = ev_model(model, df_dev, source, "dev")
    report_test, scores_test = ev_model(model, df_test, source, "test")
    report_train, scores_train = ev_model(model, df_train, source, "train")

    rows = scores_dev + scores_test + scores_train

    with open(f"predictions/lang_gs-{source}.jsonl", "w") as f:
        f.write("\n".join([json.dumps(r) for r in rows]))

    with open(f"reports/lang_gs-{source}.txt", "w") as f:
        f.write(
            f"TRAIN\n\n{report_train}\n\n\n\nDEV\n\n{report_dev}\n\n\n\nTEST\n\n{report_test}"
        )

    with open(f"reports/lang_gs-{source}.best.json", "w") as f:
        json.dump(model.best_params_, f)
