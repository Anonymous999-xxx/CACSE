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

After installation, make sure you download models' [checkpoint](https://drive.google.com/file/d/1aBbigGlMYq7ipy-rdSPFM9kmN4SWmdWt/view?usp=drive_link) Google Drive and copy all the folders into the directory where the project resides.

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

### The performance of the CACSE and baseline models on 18 multilingual/cross-language semantic similarity tasks on the STS22 test set

| **STS22** | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **CACSE** |
|:------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|
| **ar**    | **38.33**      | 32.48                | 34.94                | 21.08                | 33.58              | 36.08              | 35.33                | 33.36              |
| **de**    | 24.70               | **28.50**       | 24.47                | 18.02                | 2.58               | 24.99              | 24.70                | 24.85              |
| **de-en** | 13.13               | 29.80                | 33.63                | **37.03**       | 20.78              | 30.33                    | 35.51                | 26.72              |
| **de-fr** | 35.92               | 32.68                | 38.29                | 2.44                 | 25.42              | 31.45                    | **39.27**       | 37.12        |
| **de-pl** | 18.82         | 12.78                | 11.30                | -26.67               | 7.08               | 9.58                     | 5.67                 | **37.51**     |
| **en**    | 59.11               | 60.66                | 61.15          | 54.96                | 54.23              | 60.16                    | **62.46**       | 62.33              |
| **es**    | 49.23               | 52.14                | 55.03                | 49.06                | 39.98              | 55.16              | 54.96                | **56.16**     |
| **es-en** | 30.44               | 37.84                | 36.83                | **38.53**       | 21.28              | 34.14                    | 38.50                | 31.38              |
| **es-it** | 31.48               | 42.50                | 40.91                | 44.44                | 22.54        | 31.27                    | 42.16                | **44.52**     |
| **fr**    | 61.55         | 61.31                | 60.06                | 52.95                | 31.47              | 52.96                    | 65.35                | **65.78**     |
| **fr-pl** | 39.44         | **50.71**       | -5.63                | 16.90                | 16.90              | 16.90                    | 39.44                | 28.17              |
| **it**    | 54.67               | 59.89          | 57.61                | 52.94                | 27.64              | 53.46                    | 60.60                | **62.46**     |
| **pl**    | 22.79               | **26.72**       | 23.77          | 8.23                 | 6.78               | 23.42                    | 26.09                | 21.64              |
| **pl-en** | 15.44               | **36.41**       | 30.43          | 29.48                | 28.67              | 22.82                    | 33.62                | 23.94              |
| **ru**    | 15.71               | 17.87                | 24.03          | 6.77                 | 14.03              | **24.59**           | 18.89                | 23.73              |
| **tr**    | 28.09               | **31.56**       | 29.18                | 24.27                | 16.92              | 28.33                    | 28.61                | 30.51        |
| **zh**    | 46.42               | 37.76                | **48.78**       | 47.06                | 40.12        | 40.45                    | 46.38                | 38.29              |
| **zh-en** | 4.82                | 9.87                 | 13.14                | **27.61**       | 15.06              | 11.94                    | 8.6                  | 21.41              |
| **Avg.**  | 32.78               | 36.75                | 34.33                | 26.48                | 23.61              | 32.67                    | 37.01                | **37.22**     |


### The performance of the CACSE and baseline models on 20 text classification tasks

| **Tasks(Acc)**                                                | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhinenedCSE** | **RankCSE** | **CACSE** | **+UC D** |
|:----------------------------------------------------------------------:|:-----------------:|:------------------:|:------------------:|:------------------:|:----------------:|:----------------------:|:------------------:|:------------------:|:------------------:|
| **AngryTweetsClassification**                                 | 42.30             | 40.45              | 42.57              | 41.72              | **44.35**   | 41.33                  | 42.28              | 41.22              | 41.35              |
| **Banking77Classification**                                   | 74.43             | 73.87              | 76.09              | **78.17**     | 65.88            | 75.59                  | 75.69              | 75.38              | 77.34              |
| **CUADAffiliateLicenseLicensorLegalBenchClassification**      | 78.41             | 75.00              | **85.23**     | 76.14              | 70.45            | 81.82                  | 72.73              | 73.86              | 73.86              |
| **CUADAntiAssignmentLegalBenchClassification**                | **84.73**    | 84.64              | 80.89              | 82.00              | 79.10            | 80.46                  | 83.11              | 82.17              | 81.74              |
| **CUADIrrevocableOrPerpetualLicenseLegalBenchClassification** | 83.93             | 80.36              | **85.00**     | 85.71              | 83.93            | 80.36                  | 83.57              | 79.29              | 82.86              |
| **CUADMostFavoredNationLegalBenchClassification**             | 78.13             | **84.38**     | 75.00              | 75.00              | 64.06            | 76.56                  | 71.88              | 82.81              | 79.69              |
| **EstonianValenceClassification**                             | 28.15             | 27.41              | **28.81**     | 28.24              | 28.68            | 28.53                  | 28.34              | 28.44              | 28.24              |
| **FinToxicityClassification**                                 | 44.07             | 44.84              | 45.81              | 46.47              | 42.88            | 45.53                  | 44.77              | **47.10**     | **47.11**     |
| **FrenkEnClassification**                                     | **60.84**    | 59.00              | 59.32              | 59.64              | 59.50            | 58.40                  | 60.80              | 59.75              | 59.76              |
| **FrenkSlClassification**                                     | 55.33             | 56.11              | 55.52              | 55.53              | **57.23**   | 56.00                  | 56.06              | 54.82              | 55.70              |
| **HebrewSentimentAnalysis**                                   | 51.67             | 52.43              | 52.42              | 51.81              | 52.83            | 51.13                  | **53.39**     | 50.34              | 51.05              |
| **IndonesianIdClickbaitClassification**                       | 54.26             | 53.89              | 54.09              | 54.56              | **57.57**   | 54.15                  | 53.44              | 55.53              | 55.93              |
| **LearnedHandsEducationLegalBenchClassification**             | 75.00             | 76.79              | 76.79              | 75.00              | 69.64            | 73.21                  | 78.57              | **80.36**     | **82.14**     |
| **NaijaSenti**                                                | 39.80             | 38.90              | 39.63              | 39.75              | 39.09            | 39.89                  | **40.23**     | 39.55              | 39.65              |
| **OnlineStoreReviewSentimentClassification**                  | **28.22**    | 26.71              | 27.07              | 27.28              | 26.72            | 27.34                  | 27.25              | 27.22              | 27.07              |
| **OPP115DoNotTrackLegalBenchClassification**                  | 81.82             | 78.18              | 78.18              | 90.91              | 80.91            | 80.00                  | 81.82              | **92.73**     | **93.64**     |
| **OralArgumentQuestionPurposeLegalBenchClassification**       | 22.44             | 15.71              | 21.47              | 19.87              | **24.04**   | 23.08                  | 21.79              | 22.44              | 22.76              |
| **PolEmo2**                                                   | 34.29             | 35.02              | **37.02**     | 35.59              | 36.88            | 34.64                  | 34.23              | 32.00              | 33.04              |
| **SCDBPVerificationLegalBenchClassification**                 | 58.31             | 58.31              | 61.21              | 59.37              | 54.09            | **62.27**         | 60.42              | 59.89              | 59.89              |
| **WRIMEClassification**                                       | 20.02             | 20.25              | 20.41              | 20.68              | **22.11**   | 20.53                  | 20.11              | 20.18              | 20.03              |
| **Avg.**                                                      | 54.81             | 54.11              | 55.13              | 55.17              | 53.00            | 54.54                  | 54.52              | **55.25**     | **55.64**     |

### The performance of the CACSE and baseline models on 20 text retrieval tasks

| **Tasks(Map@10)**                   | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **CACSE** | **+UC D** |
|:--------------------------------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:------------------:|
| **ArguAna**                         | 33.58               | 33.46                | 31.36                | 33.66                | 31.09              | 31.25                    | 29.91                | 32.79              | **35.70**     |
| **BSARDRetrieval**                  | 0.91                | 1.10                 | 1.46                 | 0.54                 | 0.73               | 1.70                     | 1.52                 | **1.54**      | 0.70               |
| **CQADupstackPhysicsRetrieval**     | 17.72               | 16.74                | 18.70                | **20.29**       | 16.01              | 18.56                    | 16.52                | 17.25              | 20.01              |
| **CQADupstackProgrammersRetrieval** | 12.76               | 12.10                | 14.09                | 15.77                | 11.34              | 14.43                    | 11.28                | 12.83              | **15.81**     |
| **CQADupstackStatsRetrieval**       | 9.42                | 9.70                 | 9.96                 | **10.55**       | 8.81               | 9.98                     | 6.98                 | 7.92               | 10.26              |
| **CQADupstackWebmastersRetrieval**  | 13.32               | 13.41                | 14.16                | 16.00                | 13.49              | 14.89                    | 12.20                | 13.92              | **15.96**     |
| **FiQA-PL**                         | 0.52                | 0.59                 | 0.41                 | **0.62**        | 0.16               | 0.33                     | 0.35                 | 0.56               | 0.34               |
| **JaQuADRetrieval**                 | 3.00                | 3.16                 | 2.72                 | 4.01                 | 3.21               | 3.27                     | 3.89                 | **4.13**      | 3.89               |
| **Ko-miracl**                       | 0.61                | 0.55                 | 1.00                 | 0.96                 | 0.55               | **1.45**            | 0.55                 | 0.55               | 0.66               |
| **LegalBenchConsumerContractsQA**   | 40.13               | 42.92                | 33.74                | 33.22                | 35.06              | 30.79                    | 42.14                | **43.26**     | 42.50              |
| **LegalBenchCorporateLobbying**     | 77.23               | 78.34                | 77.50                | 78.02                | 74.84              | 74.41                    | 69.61                | 75.80              | **79.23**     |
| **LegalSummarization**              | 45.57               | 45.80                | 44.45                | 46.20                | 43.43              | 43.29                    | 44.25                | 46.44              | **48.84**     |
| **LEMBQMSumRetrieval**              | 9.64                | 11.54                | 10.10                | 10.82                | 9.47               | 9.09                     | 10.84                | **13.06**     | 12.41              |
| **LEMBSummScreenFDRetrieval**       | 39.88               | 45.34                | 41.75                | 38.93                | 28.58              | 41.46                    | 40.40                | **48.30**     | 44.27              |
| **MedicalQARetrieval**              | 20.25               | 18.06                | 17.35                | **27.73**       | 17.66              | 22.09                    | 19.20                | 20.00              | 23.01              |
| **NFCorpus**                        | 3.66                | 3.55                 | 3.43                 | **4.44**        | 2.51               | 3.81                     | 2.10                 | 3.43               | 3.96               |
| **QuoraRetrieval**                  | 75.58               | 74.23                | 75.66                | 77.57                | 70.06              | 74.54                    | 76.41                | 76.12              | **77.98**     |
| **Touche2020**                      | 3.60                | 1.86                 | 2.58                 | 1.97                 | 1.72               | **2.83**            | 2.67                 | 2.50               | 2.70               |
| **TRECCOVID**                       | 0.64                | 0.56                 | **0.65**        | 0.63                 | 0.56               | 0.61                     | 0.48                 | 0.47               | 0.61               |
| **VieQuADRetrieval**                | 1.46                | 3.87                 | 1.86                 | 1.39                 | 1.77               | 1.28                     | 2.34                 | **3.39**      | 2.51               |
| **Avg.**                            | 20.47               | 20.84                | 20.15                | 21.17                | 18.55              | 20.00                    | 19.68                | **21.21**     | **22.07**     |

### The performance of the CACSE and baseline models on 20 text reranking tasks

| **Task(Map)**                 | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **CACSE** | **+UC D** |
|:--------------------------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:------------------:|
| **AlloprofReranking**         | 30.46               | **32.74**       | 28.33                | 27.97                | 25.31              | 27.34                    | 29.25                | 31.40              | 30.04              |
| **AskUbuntuDupQuestions**     | 51.88               | 52.28                | 52.08                | 52.83                | 45.53              | 51.60                    | **53.76**       | 51.41              | 52.66              |
| **CMedQAv1-reranking**        | 13.07               | 13.63                | 14.05                | **17.23**       | 11.01              | 14.61                    | 14.04                | 13.86              | 15.85              |
| **CMedQAv2-reranking**        | 13.97               | 14.78                | 15.26                | **17.21**       | 11.69              | 15.06                    | 14.47                | 14.79              | 15.74              |
| **MindSmallReranking**        | 28.68               | 28.86                | 29.34                | 29.18                | 26.14              | 28.10                    | **29.46**       | 28.29              | 28.86              |
| **MIRACLReranking**           | 51.71               | **52.58**       | 51.67                | 52.18                | 49.97              | 52.10                    | 52.21                | 52.27              | 52.43              |
| **MMarcoReranking**           | 2.48                | 3.77                 | 3.64                 | **4.96**        | 2.70               | 4.02                     | 3.34                 | 2.92               | 4.64               |
| **SciDocsRR**                 | 67.87               | 70.48                | 70.37                | 71.29                | 58.90              | 67.63                    | 69.89                | 70.28              | **71.64**     |
| **StackOverflowDupQuestions** | 39.57               | 40.64                | 42.77                | **44.21**       | 31.06              | 42.64                    | 41.18                | 40.80              | 43.32              |
| **SyntecReranking**           | 56.53               | **58.92**       | 51.37                | 52.87                | 49.25              | 53.98                    | 52.43                | 54.53              | 55.32              |
| **T2Reranking**               | 55.20               | 55.87                | **56.27**       | 56.71                | 52.10              | 56.16                    | 55.59                | 55.54              | 56.28              |
| **Avg.**                      | 37.40               | 38.60                | 37.74                | 38.78                | 33.06              | 37.57                    | 37.78                | 37.83              | **38.80**     |

### We use the SentEval package for evaluation

| **Model**               | **Transfer tasks(8 Avg.)** | **Probing Tasks(10 Avg.)** |
|:--------------------------------:|:-----------------------------------:|:-----------------------------------:|
| **Glove**               | 81.45                               | 55.83                               |
| **Crawl**               | 82.75                               | 62.65                               |
| **Skip-thought** | 83.50                               | -                                   |
| **IS-BERT**      | 85.83                               | -                                   |
| **SimCSE-BERT**              | 85.81                               | 68.10                               |
| **CACSE-BERT**               | 85.66                               | **69.40**                      |
| **CACSE-BERT-Distilled**     | **85.90**                      | 69.33                               |
| **SimCSE-RoBERTa**              | 84.84                               | 63.85                               |
| **CACSE-RoBERTa**               | 85.10                               | 65.64                               |
| **CACSE-RoBERTa-Distilled**     | **86.77**                      | **66.22**                      |

