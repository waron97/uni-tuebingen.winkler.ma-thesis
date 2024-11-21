import json
import random


l = []
with open("./data/subtaskA_train_monolingual.jsonl", "r") as f:
    for line in f:
        try:
            j = json.loads(line)
            l.append(j)
        except:
            pass

max_id = max([r["id"] for r in l]) + 1

with open("./data/subtaskA_dev_monolingual.jsonl", "r") as f:
    for i, line in enumerate(f):
        try:
            j = json.loads(line)
            l.append({**j, "id": max_id + j["id"]})
        except:
            pass

random.shuffle(l)

models = ["chatGPT", "cohere", "davinci", "dolly", "human", "bloomz"]

train = []
dev = []

amount = 3000
amount_bloomz = 1000
amount_human = (4 * amount) + amount_bloomz

counters = {model: 0 for model in models}


for record in l:
    m = record["model"]
    limit = amount
    if m == "human":
        limit = amount_human
    if m == "bloomz":
        limit = amount_bloomz
    if counters[m] < limit:
        dev.append(record)
        counters[m] += 1
    else:
        train.append(record)

with open("../task/custom-train.jsonl", "w") as f:
    f.write("\n".join([json.dumps(record) for record in train]))
with open("../task/custom-dev.jsonl", "w") as f:
    f.write("\n".join([json.dumps(record) for record in dev]))
