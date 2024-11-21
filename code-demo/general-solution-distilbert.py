from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
import json
import os

base = os.path.join(".", "data")
feats_dev = os.path.join(base, "lang-feats-dev.jsonl")
feats_train = os.path.join(base, "lang-feats-train.jsonl")
feats_test = os.path.join(base, "lang-feats-test.jsonl")

data_dev = os.path.join(base, "subtaskA_dev_monolingual.jsonl")
data_train = os.path.join(base, "subtaskA_train_monolingual.jsonl")
data_test = os.path.join(base, "subtaskA_monolingual.jsonl")

def get_df(split, do_feats=False):
    p1, p2 = "", ""
    feats, labels, texts, models = {}, {}, {}, {}
    if (split == "train"):
        p1, p2 = data_train, feats_train
    elif (split == "dev"):
        p1, p2 = data_dev, feats_dev
    elif (split == "test"):
        p1, p2 = data_test, feats_test

    with open(p1, "r") as f:
        for line in f:
            try:
                j = json.loads(line)
                labels[j["id"]] = j["label"]
                texts[j["id"]] = j.get("text", None)
                models[j["id"]] = j.get("model", None)
            except:
                pass

    if do_feats:
        with open(p2, "r") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    feats[j["id"]] = j
                except:
                    pass

    d = []

    for _id in labels:
        r = {"label": labels[_id], "text": texts[_id],
             "model": models[_id]}
        if do_feats:
            r = {**r, **feats[_id]}

        d.append(r)

    df = pd.DataFrame(d)
    return df

accuracy = evaluate.load("accuracy")
checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

df_train = Dataset.from_pandas(get_df("train")[["text", "label"]])
df_dev = Dataset.from_pandas(get_df("dev")[["text", "label"]])
df_test = Dataset.from_pandas(get_df("test")[["text", "label"]])

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenized_train = df_train.map(preprocess_function, batched=True)
tokenized_dev = df_dev.map(preprocess_function, batched=True)
tokenized_test = df_test.map(preprocess_function, batched=True)

id2label = {0: 0, 1: 1}
label2id = {0: 0, 1: 1}

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_args = TrainingArguments(
    output_dir="finetune-out",
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
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
