# Introduction

This repository pertains to the conference paper titled "CACSE: Cross-Attention based Unsupervised Contrastive Learning for Sentence Embedding".

CACSE is a BERT-like model for computing sentence embedding vectors, trained using unsupervised contrastive learning.

# How to Use

We recommend you train CACSE with RAM >= 48GB and GPU memory >= 24GB.

## Installation
You also need to make sure your python >= 3.6 and install py repositories in requirements.txt :
```bash
pip install -r requirements.txt
```

After installation, make sure you download models' [checkpoint](https://drive.google.com/file/d/1JZNP2i8NfLmg-w-6H4zY3cToBWAKqW48/view?usp=sharing) Google Drive and copy all the folders into the directory where the project resides.

**Our code is based on [SimCSE-main](https://github.com/princeton-nlp/SimCSE) and [SimCSE-master](https://github.com/yangjianxin1/SimCSE) replication, which is gratefully acknowledged.**

For CACSE-BERT and CACSE-RoBERTa, our chosen unsupervised checkpoint are [InfoCSE-base](https://huggingface.co/ffgcc/InfoCSE-bert-base) and [ESimCSE-base](https://huggingface.co/ffgcc/esimcse-bert-base-uncased), respectively. Since the constraint layer of the CACSE-BERT and CACSE-RoBERTa cross-attention cascade are implemented by the encoder layer of the BERT-base and RoBERTa-base, it is necessary to download the pre-trained weights of the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) and [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) first.

Please download all the files of the above four models to the corresponding directories:

> [InfoCSE-base](https://huggingface.co/ffgcc/InfoCSE-bert-base) to ffgccInfoCSE-bert-base;
> 
> [ESimCSE-base](https://huggingface.co/ffgcc/esimcse-bert-base-uncased) to ffgccesimcse-base;
> 
> [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) to bert-base;
> 
> [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) to Roberta-base.

## Train
### Train CACSE-BERT
```bash
python train.py
```

### Train CACSE_distill(BERT)
```bash
python CACSE_distilled.py
```

### Train CACSE+UC_distill(BERT)
```bash
python CACSE+UC_distil.py
```

At the completion of the training of the above three files, under the save path, we re- save all the saved checkpoints in Huggingface format, and the two trained Submodels in CACSE-BERT are saved separately in Huggingface format.

## Evaluation
We use the [SentEval](https://github.com/facebookresearch/SentEval) package to evaluate the sentence embedding quality of CACSE, before evaluation, please download the relevant dataset with the following command:(**For convenience, we have already downloaded and uploaded the dataset without any further download operations.**)

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

### Eval CACSE-BERT

Unlike SimCSE, we have integrated the corresponding evaluation module into **train&#46;py**, which determines whether CACSE is training and then evaluating(True or 1) or directly evaluating(False or 0) by specifying the hyperparameter **args.do_train**.

**However, before the first evaluation, run train&#46;py with the default parameters to train CACSE-BERT, to save the model weights for CACSE-BERT.**

We have described the model evaluation file <a href="#test1">here</a>, if you want to evaluate directly without training, set the hyperparameter **arg.do_train** to **0** or **False**, provided that the CACSE-BERT weights have been saved.

For CACSE+UC evaluation and CACSE+UC_distill training and evaluation, please first to download the [unsupervised checkpoint](https://huggingface.co/ffgcc/InfoCSE-bert-base) and specify the path **args.path_to_UC**.

### Eval CACSE-RoBERTa
```bash
python eval_CACSE_RoBERTa.py
```

# Evaluation Result

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STSB | SICKR | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| <table><tr><td colspan="2" align="center">BERT-base</td></tr></table> |
| CACSE | 75.38 | 83.23 | 76.45 | 84.14 | 79.66 | 81.27 | 72.98 | 79.02 |
| CACSE D | 74.75 | 83.74 | 76.23 | 83.87 | 79.58 | 81.18 | 73.58 | 78.99 |
| CACSE+UC♠  | 75.57 | 85.15 | 78.27 | 86.01 | 81.85 | 83.13 | 73.32 | 80.47 |
| CACSE+UC D♠ | 74.88 | 85.18 | 78.06 | 85.59 | 81.40 | 82.57 | 73.41 | 80.16 |
| <table><tr><td colspan="2" align="center">RoBERTa-base</td></tr></table> |
| CACSE | 75.38 | 83.23 | 76.45 | 84.14 | 79.66 | 81.27 | 72.98 | 79.02 |
| CACSE D | 74.75 | 83.74 | 76.23 | 83.87 | 79.58 | 81.18 | 73.58 | 78.99 |
| CACSE+UC♠  | 75.57 | 85.15 | 78.27 | 86.01 | 81.85 | 83.13 | 73.32 | 80.47 |
| CACSE+UC D♠ | 74.88 | 85.18 | 78.06 | 85.59 | 81.40 | 82.57 | 73.41 | 80.16 |

# Generalization of BERT-like model to [LLaMA2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

This section uses CACSE-BERT's knowledge of the STS-B and SICKR datasets to generalize to LLaMA2-7B through instruction fine-tuning.
See the paper for more details, and this section gives an implementation description.

## Finetune 

We use [alpaca-lora](https://github.com/tloen/alpaca-lora) to 
fine-tune llama, first we make the corresponding dataset and then follow the fine-tuning commands of 
alpaca(python finetune.py \...) to get the LoRA weights file.

We put the weights file in **LoRA_weight**.

## Export Huggingface checkpoint

Next, we use [alpaca-lora](https://github.com/tloen/alpaca-lora)'s **export_hf_checkpoint.py** to fuse the lora weights with 
LLaMA2-7B to get the fine tuned model.

## Evaluation of LLM

We used [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate both the original and the LLaMA2 that we fine-tuned.

# Citing
MTEB was introduced in "[CACSE: Cross-Attention based Unsupervised Contrastive Learning for Sentence Embedding](https://arxiv.org/abs/xxx)", feel free to cite:

```bibtex
@article{xxx}
```
