# some code borrowed from https://github.com/liuzywen/TriTransNet

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, tgt, memory):
        mixed_query_layer = self.query(tgt)
        mixed_key_layer = self.key(memory)
        mixed_value_layer = self.value(memory)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None

        features = x
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class TransformerEncoderLayer(nn.Module):

    def __init__(self, config, vis):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[ROOT + '/' + ATTENTION_Q + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            key_weight = np2th(weights[ROOT + '/' + ATTENTION_K + "/kernel"]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            value_weight = np2th(weights[ROOT + '/' + ATTENTION_V + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            out_weight = np2th(weights[ROOT + '/' + ATTENTION_OUT + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()

            query_bias = np2th(weights[ROOT + '/' + ATTENTION_Q + "/bias"]).view(-1)
            key_bias = np2th(weights[ROOT + '/' + ATTENTION_K + "/bias"]).view(-1)
            value_bias = np2th(weights[ROOT + '/' + ATTENTION_V + "/bias"]).view(-1)
            out_bias = np2th(weights[ROOT + '/' + ATTENTION_OUT + "/bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + '/' + FC_0 + "/kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + '/' + FC_1 + "/kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + '/' + FC_0 + "/bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + '/' + FC_1 + "/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + '/' + MLP_NORM + "/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + '/' + MLP_NORM + "/bias"]))


class TransformerEncoderLayer_use_fg(nn.Module):

    def __init__(self, config, vis):
        super(TransformerEncoderLayer_use_fg, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_mg = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.attn_mg = Attention(config, vis)

        self.dropout = Dropout(config.transformer["dropout_rate"])

        self.sigmoid = nn.Sigmoid()
        self.fg = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forget_gate(self, features, pre_features):
        forget_mask = self.fg(torch.cat((features, pre_features), 2))
        forget_mask = self.sigmoid(forget_mask)
        forget_mask = self.dropout(forget_mask)
        features = forget_mask.mul(features)
        return features

    def forward(self, x, y):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x)
        x = x + h

        h = x
        x = self.attention_norm_mg(x)
        y = self.attention_norm_mg(y)
        x, weights = self.attn_mg(x, y)
        x = self.forget_gate(x, h)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[ROOT + '/' + ATTENTION_Q + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            key_weight = np2th(weights[ROOT + '/' + ATTENTION_K + "/kernel"]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            value_weight = np2th(weights[ROOT + '/' + ATTENTION_V + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
            out_weight = np2th(weights[ROOT + '/' + ATTENTION_OUT + "/kernel"]).view(self.hidden_size,
                                                                                     self.hidden_size).t()

            query_bias = np2th(weights[ROOT + '/' + ATTENTION_Q + "/bias"]).view(-1)
            key_bias = np2th(weights[ROOT + '/' + ATTENTION_K + "/bias"]).view(-1)
            value_bias = np2th(weights[ROOT + '/' + ATTENTION_V + "/bias"]).view(-1)
            out_bias = np2th(weights[ROOT + '/' + ATTENTION_OUT + "/bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            self.attn_mg.query.weight.copy_(query_weight)
            self.attn_mg.key.weight.copy_(key_weight)
            self.attn_mg.value.weight.copy_(value_weight)
            self.attn_mg.out.weight.copy_(out_weight)
            self.attn_mg.query.bias.copy_(query_bias)
            self.attn_mg.key.bias.copy_(key_bias)
            self.attn_mg.value.bias.copy_(value_bias)
            self.attn_mg.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + '/' + FC_0 + "/kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + '/' + FC_1 + "/kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + '/' + FC_0 + "/bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + '/' + FC_1 + "/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/bias"]))
            self.attention_norm_mg.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/scale"]))
            self.attention_norm_mg.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + "/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + '/' + MLP_NORM + "/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + '/' + MLP_NORM + "/bias"]))


class UnionTransformer(nn.Module):
    def __init__(self, config, vis):
        super(UnionTransformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(2):
            layer = TransformerEncoderLayer(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, tgt):
        attn_weights = []
        output = tgt
        for layer_block in self.layer:
            output, weights = layer_block(output)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(output)
        return encoded, attn_weights


class MutualGuidance(nn.Module):
    def __init__(self, config, vis):
        super(MutualGuidance, self).__init__()
        self.vis = vis
        self.layer_s = nn.ModuleList()
        self.layer_t = nn.ModuleList()
        self.encoder_norm_s = LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder_norm_t = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(2):
            layer = TransformerEncoderLayer_use_fg(config, vis)
            self.layer_s.append(copy.deepcopy(layer))

        for _ in range(2):
            layer = TransformerEncoderLayer_use_fg(config, vis)
            self.layer_t.append(copy.deepcopy(layer))

    def forward(self, tgt, memory):
        output = tgt
        for layer_block_s, layer_block_t in zip(self.layer_s, self.layer_t):
            ori = output
            output = layer_block_s(output, memory)
            memory = layer_block_t(memory, ori)

        x = self.encoder_norm_s(output)
        y = self.encoder_norm_t(memory)
        return x, y


class Transformer_MG(nn.Module):
    def __init__(self, config, img_size, in_channels, vis):
        super(Transformer_MG, self).__init__()
        self.embeddings_rgb = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.embeddings_flow = Embeddings(config, img_size=img_size, in_channels=in_channels)
        self.MG = MutualGuidance(config, vis)
        self.UnionTrans = UnionTransformer(config, vis)

        self.fg = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def encode_rgb(self, img):
        embedding_output, features = self.embeddings_rgb(img)
        return embedding_output, features

    def encode_flo(self, flow):
        embedding_output, features = self.embeddings_flow(flow)
        return embedding_output, features

    def Union(self, encoded_feat):
        decoded_mask_enc, attn_weights = self.UnionTrans(encoded_feat)

        return decoded_mask_enc, attn_weights

    def forward(self, rgb, flow):
        encoded_rgb, _ = self.encode_rgb(rgb)
        encoded_flo, _ = self.encode_flo(flow)

        rgb_bi, flo_bi = self.MG(encoded_rgb, encoded_flo)

        final_feat, _ = self.Union(self.fg(torch.cat([rgb_bi, flo_bi], 2)))

        return final_feat


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Reshape(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.config = config
        head_channels = in_channels
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        output = self.conv_more(x)

        return output


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=28, in_channels=64, num_classes=1, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()

        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_MG(config, img_size, in_channels, vis)
        self.reshape = Reshape(config, in_channels)
        self.config = config

    def forward(self, rgb, flow):
        x = self.transformer(rgb, flow)
        x = self.reshape(x)

        return x

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights

            self.transformer.MG.encoder_norm_s.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.MG.encoder_norm_s.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            self.transformer.MG.encoder_norm_t.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.MG.encoder_norm_t.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            self.transformer.UnionTrans.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.UnionTrans.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new_e = self.transformer.embeddings_rgb.position_embeddings
            posemb_new_d = self.transformer.embeddings_flow.position_embeddings
            if posemb.size() == posemb_new_e.size():
                self.transformer.embeddings_rgb.position_embeddings.copy_(posemb)
                self.transformer.embeddings_flow.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new_e.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings_rgb.position_embeddings.copy_(posemb)
                self.transformer.embeddings_flow.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new_e.size()))
                ntok_new = posemb_new_e.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings_rgb.position_embeddings.copy_(np2th(posemb))
                self.transformer.embeddings_flow.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.MG.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            for bname, block in self.transformer.UnionTrans.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings_rgb.hybrid:
                self.transformer.embeddings_rgb.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                self.transformer.embeddings_flow.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings_rgb.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings_flow.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings_rgb.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

                for bname, block in self.transformer.embeddings_flow.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
