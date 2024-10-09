# Oral Exam Papers

## Information retrieval

### Proposal 1

[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

Publication Date: 2020, ColBERT v2, then released in 2023, but this original paper explains the intuition more concisely

Google Scholar citations: 1193

*The paper presents a full system using fine-tuned BERT to perform retrieval.*
*What makes this paper stand out is that it talks about quantization, indexing, and different aspects of a modern retrieval system.*

### Proposal 2

[Query2doc: Query Expansion with Large Language Models](https://arxiv.org/abs/2303.07678)

Publication Date: 2023

Google Scholar citations: 124

*This study deals with the strategy of query expansion.*
*When a query is submitted to the retriever, many systems intercept it and extend it in some way, by adding context, or pre-generated answers.*
*In this case, LLMs are used to extend the query with a passage, in an attempt to help the IR system to match relevant documents.*
*They report improvements on classical retrievers like BM25, but not on neural-native systems, which likely wouldn't benefit very much from this.*

### Proposal 3

[Fine-tuning LLaMA for multi-stage text retrieval](https://dl.acm.org/doi/abs/10.1145/3626772.3657951)

Publication Date: 2024

Google Scholar citations: 73

*This paper presents 2 interesting systems: one similar to ColBERT, described in the first paper, a dual-model retriever using LLaMA.*
*It also presents a pointwise ranker, a type of retrieval model that evaluates query and document together, usually only for a subset of pre-retrieved documents.*

### Proposal 4

[Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP](https://arxiv.org/abs/2212.14024)

Publication Date: 2022

Google Scholar citations: 158

### Proposal 5

[Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118)

Publication date: 2021

Google Scholar citations: 477

*This study presents an unsupervised retriever using contrastive learning.*
*The training objective is to determine if two passages from the corpus belong to the same document or not.*
*They evaluate on languages with very little available data, which is what drew me to the study.*

### Proposal 6

[Learning to Filter Context for Retrieval-Augmented Generation](https://arxiv.org/abs/2311.08377)

Publication Date: 2023

Google Scholar citations: 49

*This paper talks about active summarization of documents during retrieval.*
*This operation apparently improves results, but might also be an important tool for cost management, since language models accessed over paid APIs do not mix well with large contexts.*
*To be honest, I wanted to find some non-LM stuff for at least one paper, but there doesn't seem to be much done aside from them in recent years, so that search is as of right now unfruitful.*
*If I happen upon something, this is the paper I'd most likely drop.*

## Machine translation


