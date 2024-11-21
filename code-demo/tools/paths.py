import os


base = os.path.join(".", "data")

custom_train = os.path.join(base, "custom-train.jsonl")
custom_dev = os.path.join(base, "custom-dev.jsonl")

feats_dev = os.path.join(base, "lang-feats-dev.jsonl")
feats_train = os.path.join(base, "lang-feats-train.jsonl")
feats_test = os.path.join(base, "lang-feats-test.jsonl")
feats_joined = os.path.join(base, "lang-feats-joined.jsonl")

data_dev = os.path.join(base, "subtaskA_dev_monolingual.jsonl")
data_train = os.path.join(base, "subtaskA_train_monolingual.jsonl")
data_test = os.path.join(base, "subtaskA_monolingual.jsonl")
