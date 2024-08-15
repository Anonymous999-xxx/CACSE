import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class constraint_layer_setting(nn.Module):
    def __init__(self, pretrained_model, constraint_trainstate_pos, dropout=0.1):
        super(constraint_layer_setting, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout

        self.Encoderlayers_Model = AutoModel.from_pretrained(pretrained_model, config=config)

        for param in self.Encoderlayers_Model.encoder.layer.parameters():
            param.requires_grad = False

        constraint_trainstate_pos = -1 * constraint_trainstate_pos
        for param in self.Encoderlayers_Model.encoder.layer[constraint_trainstate_pos:].parameters():
            param.requires_grad = True
        self.constraint_layer = self.Encoderlayers_Model.encoder.layer[constraint_trainstate_pos:]

    def forward(self, inputA, inputB):
        inputA, inputB = torch.transpose(inputA, 0, 1), torch.transpose(inputB, 0, 1)

        for encoder in self.constraint_layer:
            inputA, inputB = encoder(inputA)[0], encoder(inputB)[0]

        inputA, inputB = torch.transpose(inputA, 0, 1), torch.transpose(inputB, 0, 1)

        return inputA, inputB


def devided(a):
    dim0 = a.size(0)
    input_size = a.size()
    # print("asize",asize)
    b, c = [], []

    for i in range(dim0):
        if i % 2 == 0:
            b.append(a[i])
        else:
            c.append(a[i])

    b = torch.cat(b, dim=0)
    c = torch.cat(c, dim=0)

    b = b.reshape(-1, input_size[-2], input_size[-1])
    c = c.reshape(-1, input_size[-2], input_size[-1])

    return b, c


class firstlayer_cascade(nn.Module):
    def __init__(self, num_head, hidden_dim, constraint_trainstate_pos, constraint_layer_pretrained):
        super(firstlayer_cascade, self).__init__()

        self.W_A = nn.Linear(hidden_dim, hidden_dim)
        self.W_B = nn.Linear(hidden_dim, hidden_dim)

        self.selfatt_A = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)
        self.selfatt_B = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)

        self.crossatt_A = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)
        self.crossatt_B = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)

        self.bl_A = nn.LayerNorm(normalized_shape=hidden_dim)
        self.bl_B = nn.LayerNorm(normalized_shape=hidden_dim)

        self.constraint_layer = constraint_layer_setting(pretrained_model=constraint_layer_pretrained,
                                                         constraint_trainstate_pos=constraint_trainstate_pos)

        self.ln_A = nn.LayerNorm(normalized_shape=hidden_dim)
        self.ln_B = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_hidden_state):
        input_devid_A, input_devid_B = devided(input_hidden_state)

        input_devid_A, input_devid_B = input_devid_A[:, 0].unsqueeze(1), input_devid_B[:, 0].unsqueeze(1)
        input_devid_A = self.W_A(input_devid_A) + input_devid_A
        # input_devid_A = self.W_A(input_devid_A)
        input_devid_A = self.bl_A(input_devid_A)

        input_devid_A = torch.transpose(input_devid_A, 0, 1)
        #         48 64 768
        input_devid_A, _ = self.selfatt_A(input_devid_A, input_devid_A, input_devid_A)
        #####
        input_devid_B = self.W_B(input_devid_B) + input_devid_B
        # input_devid_B = self.W_B(input_devid_B)
        input_devid_B = self.bl_B(input_devid_B)

        input_devid_B = torch.transpose(input_devid_B, 0, 1)
        #         48 64 768
        input_devid_B, _ = self.selfatt_B(input_devid_B, input_devid_B, input_devid_B)
        #####
        crossout_A, _ = self.crossatt_A(input_devid_A, input_devid_A, input_devid_B)
        crossout_B, _ = self.crossatt_B(input_devid_B, input_devid_B, input_devid_A)

        crossout_A = self.ln_A(torch.mul(crossout_A, input_devid_A))
        crossout_B = self.ln_B(torch.mul(crossout_B, input_devid_B))
        # 48 64 768
        #####
        out_A = self.dropout(crossout_A)
        out_B = self.dropout(crossout_B)

        out_A = torch.transpose(out_A, 0, 1)
        out_B = torch.transpose(out_B, 0, 1)

        out_A, out_B = self.constraint_layer(out_A, out_B)

        return out_A, out_B
        # 64 48 768


class Subsequentlayer_cascade(nn.Module):
    def __init__(self, num_head, hidden_dim):
        super(Subsequentlayer_cascade, self).__init__()

        self.W_A = nn.Linear(hidden_dim, hidden_dim)
        self.W_B = nn.Linear(hidden_dim, hidden_dim)

        self.selfatt_A = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)
        self.selfatt_B = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)

        self.crossatt_A = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)
        self.crossatt_B = nn.MultiheadAttention(num_heads=num_head, embed_dim=hidden_dim)

        self.bl_A = nn.LayerNorm(normalized_shape=hidden_dim)
        self.bl_B = nn.LayerNorm(normalized_shape=hidden_dim)

        self.ln_A = nn.LayerNorm(normalized_shape=hidden_dim)
        self.ln_B = nn.LayerNorm(normalized_shape=hidden_dim)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_hidden_state, beforelayer_outA, beforelayer_outB):
        #   128 48 768

        input_devid_A, input_devid_B = devided(input_hidden_state)

        #####
        input_devid_A, input_devid_B = input_devid_A[:, 0].unsqueeze(1), input_devid_B[:, 0].unsqueeze(1)
        # input_devid_A = self.W_A(input_devid_A) + input_devid_A
        input_devid_A = self.W_A(input_devid_A)
        input_devid_A = self.bl_A(input_devid_A + beforelayer_outA)
        input_devid_A = torch.transpose(input_devid_A, 0, 1)

        input_devid_A, _ = self.selfatt_A(input_devid_A, input_devid_A, input_devid_A)

        input_devid_B = self.W_B(input_devid_B)
        input_devid_B = self.bl_B(input_devid_B + beforelayer_outB)
        input_devid_B = torch.transpose(input_devid_B, 0, 1)

        input_devid_B, _ = self.selfatt_B(input_devid_B, input_devid_B, input_devid_B)

        crossout_A, _ = self.crossatt_A(input_devid_A, input_devid_A, input_devid_B)
        crossout_B, _ = self.crossatt_B(input_devid_B, input_devid_B, input_devid_A)

        crossout_A = self.dropout(crossout_A)
        crossout_B = self.dropout(crossout_B)

        crossout_A = self.ln_A(torch.mul(crossout_A, input_devid_A))
        crossout_B = self.ln_B(torch.mul(crossout_B, input_devid_B))

        out_A = torch.transpose(crossout_A, 0, 1)
        out_B = torch.transpose(crossout_B, 0, 1)

        return out_A, out_B


class cascade(nn.Module):
    def __init__(self, num_head, hidden_dim, begin_encoder_layer, constraint_trainstate_pos,
                 constraint_layer_pretrained):
        super(cascade, self).__init__()

        assert begin_encoder_layer >= 7

        self.first_layer = firstlayer_cascade(num_head=num_head, hidden_dim=hidden_dim,
                                              constraint_trainstate_pos=constraint_trainstate_pos,
                                              constraint_layer_pretrained=constraint_layer_pretrained)
        self.begin_bertlayer = begin_encoder_layer
        if begin_encoder_layer == 7:
            self.Subsequentlayer = nn.ModuleDict(
                {"index1": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index2": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index3": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index4": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index5": Subsequentlayer_cascade(num_head, hidden_dim)
                 })
        elif begin_encoder_layer == 8:
            self.Subsequentlayer = nn.ModuleDict(
                {"index1": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index2": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index3": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index4": Subsequentlayer_cascade(num_head, hidden_dim),
                 })
        elif begin_encoder_layer == 9:
            self.Subsequentlayer = nn.ModuleDict(
                {"index1": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index2": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index3": Subsequentlayer_cascade(num_head, hidden_dim),
                 })
        elif begin_encoder_layer == 10:
            self.Subsequentlayer = nn.ModuleDict(
                {"index1": Subsequentlayer_cascade(num_head, hidden_dim),
                 "index2": Subsequentlayer_cascade(num_head, hidden_dim),
                 })
        elif begin_encoder_layer == 11:
            self.Subsequentlayer = nn.ModuleDict(
                {"index1": Subsequentlayer_cascade(num_head, hidden_dim),
                 })
        elif begin_encoder_layer == 12:
            pass
        else:
            raise NotImplementedError

    def forward(self, input_encoderout):
        # bert每层的输出放在一个list中

        if self.begin_bertlayer == 7:
            out_A, out_B = self.first_layer(input_encoderout[0])
            out_A, out_B = self.Subsequentlayer['index1'](input_encoderout[1], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index2'](input_encoderout[2], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index3'](input_encoderout[3], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index4'](input_encoderout[4], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index5'](input_encoderout[5], out_A, out_B)

        elif self.begin_bertlayer == 8:
            out_A, out_B = self.first_layer(input_encoderout[1])
            out_A, out_B = self.Subsequentlayer['index1'](input_encoderout[2], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index2'](input_encoderout[3], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index3'](input_encoderout[4], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index4'](input_encoderout[5], out_A, out_B)

        elif self.begin_bertlayer == 9:
            out_A, out_B = self.first_layer(input_encoderout[2])
            out_A, out_B = self.Subsequentlayer['index1'](input_encoderout[3], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index2'](input_encoderout[4], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index3'](input_encoderout[5], out_A, out_B)

        elif self.begin_bertlayer == 10:
            out_A, out_B = self.first_layer(input_encoderout[3])
            out_A, out_B = self.Subsequentlayer['index1'](input_encoderout[4], out_A, out_B)
            out_A, out_B = self.Subsequentlayer['index2'](input_encoderout[5], out_A, out_B)

        elif self.begin_bertlayer == 11:
            out_A, out_B = self.first_layer(input_encoderout[4])
            out_A, out_B = self.Subsequentlayer['index1'](input_encoderout[5], out_A, out_B)

        elif self.begin_bertlayer == 12:
            out_A, out_B = self.first_layer(input_encoderout[5])

        return out_A, out_B
