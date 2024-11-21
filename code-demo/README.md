# Code Demo

This folder contains the script files used to run the experiments described in the thesis.
Since the right environment is missing, this is meant as a viewing demonstration only.
Executing these scripts requires setting up the proper environment for them, the specifications for which might be made available in the documentation at a later date.

The intended execution order is as follows:

### 1. Creating custom partitions

The experiment presented in the thesis did not use the proposed train/dev/test splits in the shared task.
`repartition.py` merges the the original train and dev splits, and outputs the two final partitions, referenced in the code as `custom-train` and `custom-dev`.

### 2. Training single-generator classifiers

The script files `split-models-[strategy].py` are responsible for training the classifiers for each generator according to the particular strategy, and then using the best models to annotate `custom-dev` and `test` partitions.

For example `split-models-distilbert.py` will produce checkpoint metadata in the `reports` folder, and write the annotated data to `predictions/bert-[bloomz/chatGPT/cohere/davinci/dolly].jsonl`.

All single-generator model-annotated data is included in the repository directly. The precomputed language features can be found at the following links (permanence not guaranteed):

- [Features for train set](https://s3.tebi.io/winkler.stuff/semeval/lang-feats-train.jsonl)
- [Features for dev set](https://s3.tebi.io/winkler.stuff/semeval/lang-feats-dev.jsonl)
- [Features for test set](https://s3.tebi.io/winkler.stuff/semeval/lang-feats-test.jsonl)
- [Joined train+dev](https://s3.tebi.io/winkler.stuff/semeval/lang-feats-joined.jsonl) (helpful after deriving custom-train and custom-dev)

### 3. Training the final classifier

The script `ensemble-nn.py` contains the logic to train the final classifiers, as well as to annotate the development and test sets with the best ensemble found. 

### 4. Generalization study

The scripts for the generalization analyses in the thesis can be found in `model-generalization.py`.