import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from cascade_model import cascade


def add_into(a, b):
    assert a.size(0) == b.size(0)
    c = []
    for i in range(a.size(0)):
        c.append(a[i].unsqueeze(0))
        c.append(b[i].unsqueeze(0))
    c = torch.cat(c, dim=0)
    return c


class CACSEModel(nn.Module):
    """
    CASCE model consists of two submodels and a cross-attention cascade
    num_head indicates the number of self-attentive and cross-attentive heads in the cascade, default is 6
    hidden_dim denotes the dimension of the hidden state vector of the BERT-base-like model, default is 768
    begin_encoder_layer refers to the encoder layer of the submodel corresponding to the first layer of the cross-attention cascade.
    constraint_trainstate_pos refers to the constraint layer included in the first layer of the cross-attention cascade, which intercepts the position of the coding layer of the pre-trained model
    constraint_layer_pretrained refers to the source of the constraint layer, BERT-base and RoBERTa-base, respectively
    """
    def __init__(self, Submodel_1_path, Submodel_2_path, num_head, hidden_dim, begin_encoder_layer,
                 constraint_trainstate_pos, constraint_layer_pretrained):
        super(CACSEModel, self).__init__()

        config_sub1 = AutoConfig.from_pretrained(Submodel_1_path)
        config_sub1.attention_probs_dropout_prob = 0.15
        config_sub1.hidden_dropout_prob = 0.15

        config_sub2 = AutoConfig.from_pretrained(Submodel_2_path)
        config_sub2.attention_probs_dropout_prob = 0.15
        config_sub2.hidden_dropout_prob = 0.15

        self.Submodel_1 = AutoModel.from_pretrained(Submodel_1_path, config=config_sub1)
        self.Submodel_2 = AutoModel.from_pretrained(Submodel_2_path, config=config_sub2)
        self.cascade = cascade(num_head, hidden_dim, begin_encoder_layer, constraint_trainstate_pos, constraint_layer_pretrained)

        for param in self.Submodel_1.encoder.layer[:begin_encoder_layer].parameters():
            param.requires_grad = False
        for param in self.Submodel_2.encoder.layer[:begin_encoder_layer].parameters():
            param.requires_grad = False
        # Turn off gradient updating for a portion of the encoding layer for both submodels

    def forward(self, input_ids, attention_mask, token_type_ids, model_state):

        out_Submodel_1 = self.Submodel_1(input_ids, attention_mask, token_type_ids, output_hidden_states=True,
                                         return_dict=True)
        out_Submodel_2 = self.Submodel_2(input_ids, attention_mask, token_type_ids, output_hidden_states=True,
                                         return_dict=True)
        if model_state == "train":
            out_list = [out_Submodel_1.hidden_states[t] + out_Submodel_2.hidden_states[t] for t in
                        range(7, len(out_Submodel_1.hidden_states))]

            out1, out2 = self.cascade(out_list)
            out_cross_add = add_into(out1, out2)
            out_cross_add_cls = out_cross_add[:, 0]
            return out_Submodel_1.last_hidden_state[:, 0], out_Submodel_2.last_hidden_state[:, 0], out_cross_add_cls  # [batch, 768]
            # When CACSE is in the training state, the hidden state vectors need to go through the cross-attention cascade
        elif model_state == "eval":
            #
            return out_Submodel_1.last_hidden_state[:, 0] + out_Submodel_2.last_hidden_state[:, 0]

class CACSE_distilled(nn.Module):
    def __init__(self, pretrained_model, pooling, dropout=0.015):
        super(CACSE_distilled, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.config.attention_probs_dropout_prob = dropout
        self.config.hidden_dropout_prob = dropout
        self.CACSE_distilled_model = AutoModel.from_pretrained(pretrained_model, config=self.config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.CACSE_distilled_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=True, return_dict=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


def InfoNCE_loss(y_pred, device, tau=0.05):
    """
    y_pred (tensor): [batch_size * 2, 768]
    """
    # Get the label corresponding to y_pred, [1, 0, 3, 2, ... , batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # The similarity matrix (diagonal matrix) is obtained by calculating the similarity in batch.
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # Set the diagonal of the similarity matrix to a very small value to eliminate the effect of its own
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # Similarity matrix divided by temperature coefficient
    sim = sim / tau
    # Calculate the cross entropy loss of the similarity matrix with y_true
    # Calculate the cross entropy, each case is calculated with the similarity score with other cases to get a score vector,
    # the purpose is to make the positive samples in this score vector have the highest score and the negative samples have the lowest score
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)

class CACSEModel_RoBERTa(nn.Module):
    def __init__(self, Submodel_1_path, Submodel_2_path, num_head, hidden_dim, begin_encoder_layer,
                 constraint_trainstate_pos, constraint_layer_pretrained):
        super(CACSEModel_RoBERTa, self).__init__()

        config_sub1 = AutoConfig.from_pretrained(Submodel_1_path)
        config_sub1.attention_probs_dropout_prob = 0.1
        config_sub1.hidden_dropout_prob = 0.1

        config_sub2 = AutoConfig.from_pretrained(Submodel_2_path)
        config_sub2.attention_probs_dropout_prob = 0.15
        config_sub2.hidden_dropout_prob = 0.15

        self.Submodel_1 = AutoModel.from_pretrained(Submodel_1_path, config=config_sub1)
        self.Submodel_2 = AutoModel.from_pretrained(Submodel_2_path, config=config_sub2)
        self.cascade = cascade(num_head, hidden_dim, begin_encoder_layer, constraint_trainstate_pos, constraint_layer_pretrained)

        for param in self.Submodel_1.encoder.layer[:begin_encoder_layer].parameters():
            param.requires_grad = False
        for param in self.Submodel_2.encoder.layer[:begin_encoder_layer].parameters():
            param.requires_grad = False
        # Turn off gradient updating for a portion of the encoding layer for both submodels

    def forward(self, input_ids, attention_mask, model_state):

        out_Submodel_1 = self.Submodel_1(input_ids, attention_mask)
        out_Submodel_2 = self.Submodel_2(input_ids, attention_mask)
        if model_state == "train":
            out_list = [out_Submodel_1.hidden_states[t] + out_Submodel_2.hidden_states[t] for t in
                        range(7, len(out_Submodel_1.hidden_states))]

            out1, out2 = self.cascade(out_list)
            out_cross_add = add_into(out1, out2)
            out_cross_add_cls = out_cross_add[:, 0]
            return out_Submodel_1.last_hidden_state[:, 0], out_Submodel_2.last_hidden_state[:, 0], out_cross_add_cls  # [batch, 768]
            # When CACSE is in the training state, the hidden state vectors need to go through the cross-attention cascade
        elif model_state == "eval":
            # For inference, it is straightforward to sum the h(xcls) of the last hidden state layer of the two submodels
            return out_Submodel_1.last_hidden_state[:, 0] + out_Submodel_2.last_hidden_state[:, 0]
