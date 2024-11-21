import json
import evaluate
import numpy as np
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
    set_seed,
    Pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from tools.data import get_df
from datasets import Dataset
import pandas as pd

# hide warning about pandas not auto casting values to new type in replace
pd.set_option("future.no_silent_downcasting", True)

SEED = 4376823
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

accuracy = evaluate.load("accuracy")
checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

train_df = get_df("custom-train")
dev_df = get_df("custom-dev")
test_df = get_df("test")

models = ["chatGPT", "cohere", "davinci", "dolly", "bloomz"]


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def ev_model(pipe: Pipeline, dataset: pd.DataFrame, split_id: str, label2id: dict):
    ds = Dataset.from_pandas(dataset)
    rows = []
    report = ""
    y_pred, y_true = [], []

    passthrough = pipe(KeyDataset(ds, "text"), truncation=True, padding=True)
    for index, out in tqdm(enumerate(passthrough), total=len(ds)):
        item = ds[index]
        scores = {i["label"]: i["score"] for i in out}
        score = scores[source]
        y_pred.append(label2id[out[0]["label"]])
        y_true.append(item["label"])
        rows.append({"split": split_id, "id": item["id"], "pred": score})

    report = classification_report(y_true, y_pred)

    return report, rows


for source in models:
    train_pos = train_df[train_df.model == source]
    train_neg = train_df[train_df.model == "human"].sample(train_pos.shape[0])

    dev_pos = dev_df[dev_df.model == source]
    dev_neg = dev_df[dev_df.model == "human"][: dev_pos.shape[0]]

    train = Dataset.from_pandas(pd.concat([train_pos, train_neg])).map(
        preprocess_function, batched=True
    )
    dev = Dataset.from_pandas(pd.concat([dev_pos, dev_neg])).map(
        preprocess_function, batched=True
    )

    id2label = {0: "human", 1: source}
    label2id = {"human": 0, source: 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir=f"runs/{source}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"best_models/{source}")
    model = DistilBertForSequenceClassification.from_pretrained(f"best_models/{source}")

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=0,
        batch_size=16,
    )

    report_train, scores_train = ev_model(pipe, train_df, "train", label2id)
    report_dev, scores_dev = ev_model(pipe, dev_df, "dev", label2id)
    report_test, scores_test = ev_model(pipe, test_df, "test", label2id)

    predictions = scores_train + scores_dev + scores_test

    with open(f"predictions/bert-{source}.jsonl", "w") as f:
        f.write("\n".join([json.dumps(r) for r in predictions]))
    with open(f"reports/bert-{source}.txt", "w") as f:
        f.write(
            f"TRAIN REPORT\n\n{report_train}\n\n\n\nDEV REPORT\n\n{report_dev}\n\n\n\nTEST REPORT\n\n{report_test}"
        )
