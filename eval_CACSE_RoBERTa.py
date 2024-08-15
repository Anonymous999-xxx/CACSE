"""
In this file we provide the code for evaluate the CACSE_RoBERTa series models, the training files will be made public after the paper is accepted.
"""

import torch
from transformers import AutoModel
from model import CACSEModel_RoBERTa
from evaluation import evaluation_roberta, eval_cacse_roberta_d, eval_CACSE_roberta_UC

CACSE_roberta = CACSEModel_RoBERTa(Submodel_1_path='Roberta-base', Submodel_2_path='Roberta-base', num_head=6,
                                   hidden_dim=768, begin_encoder_layer=11, constraint_trainstate_pos=1,
                                   constraint_layer_pretrained='Roberta-base')

CACSE_roberta.load_state_dict(torch.load('CACSE_RoBERTa_weights/CACSE_RoBERTa/pytorch_model.bin'))
CACSE_roberta.eval()
_, _ = evaluation_roberta(CACSE_roberta)
# CACSE_roberta eval result:
# ------ test ------
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | 72.67 | 83.00 | 75.69 | 84.07 | 82.01 |    82.53     |      71.92      | 78.84 |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+

CACSE_roberta_self_D = AutoModel.from_pretrained('Roberta-base')
CACSE_roberta_self_D.load_state_dict(
    torch.load('CACSE_RoBERTa_weights/CACSE_RoBERTa_self_D/pytorch_model.bin'))
CACSE_roberta_self_D.eval()
eval_cacse_roberta_d(CACSE_roberta_self_D)
# CACSE_roberta_Distill eval result:
# ------ test ------
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | 73.50 | 84.28 | 76.25 | 84.60 | 82.41 |    83.35     |      72.45      | 79.55 |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+

CACSE_roberta_UC_D = AutoModel.from_pretrained('Roberta-base')
CACSE_roberta_UC_D.load_state_dict(
    torch.load('CACSE_RoBERTa_weights/CACSE_roberta_UC_D/pytorch_model.bin'))
CACSE_roberta_UC_D.eval()
eval_cacse_roberta_d(CACSE_roberta_UC_D)
# CACSE_roberta_UC_Distill eval result:
# ------ test ------
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | 72.59 | 83.89 | 76.13 | 84.26 | 82.29 |    83.23     |      72.77      | 79.31 |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+

_, _ = eval_CACSE_roberta_UC(CACSE_roberta)
# CACSE_roberta_UC eval result:
# ------ test ------
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
# | 74.05 | 84.89 | 78.06 | 85.09 | 82.11 |    83.53     |      72.70      | 80.06 |
# +-------+-------+-------+-------+-------+--------------+-----------------+-------+
