import numpy as np
import pandas as pd

from tools.data import get_df, get_predictions_df

preds_train = get_predictions_df("custom-train")
preds_dev = get_predictions_df("custom-dev").reindex(columns=preds_train.columns)
preds_test = get_predictions_df("test").reindex(columns=preds_train.columns)

generators = ["chatGPT", "davinci", "cohere", "dolly", "bloomz"]
strategies = ["bert", "tfidf_gs", "lang_gs", "tandem"]

data = {"model": [], "strategy": [], **{g: [] for g in generators + ["human"]}}

for generator in generators:
    for strategy in strategies:
        data["model"].append(generator)
        data["strategy"].append(strategy)
        for other_generator in generators + ["human"]:
            if strategy == "tandem":
                targets = preds_dev[preds_dev.model == other_generator]
                _id1 = f"tfidf_gs-{generator}"
                _id2 = f"lang_gs-{generator}"
                scores = targets[[_id1, _id2, "id", "model", "label"]]
                labels = scores["label"].astype(int)
                preds = scores[[_id1, _id2]].map(np.round).astype(int)
                scores["ok"] = [
                    True if labels.iloc[i] in preds.iloc[i].values else False
                    for i in range(labels.shape[0])
                ]
                # print(scores)
                correct = scores[scores["ok"] == True].shape[0]
                accuracy = round(correct / scores.shape[0], 2)
                data[other_generator].append(accuracy)
            else:
                targets = preds_dev[preds_dev.model == other_generator]
                _id = f"{strategy}-{generator}"
                scores = targets[[_id, "id", "model", "label"]]
                scores["ok"] = scores["label"] == np.round(scores[_id]).astype(int)
                correct = scores[scores["ok"] == True].shape[0]
                accuracy = round(correct / scores.shape[0], 2)
                data[other_generator].append(accuracy)

results_df = pd.DataFrame(data)
print(results_df)
print(results_df.to_latex())
