#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from typing import List, Type
import os

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from .vit_backbones import CONFIGS, Transformer, VisionTransformer, np2th

MODEL_ZOO = {
    "sup_vits16_imagenet21k": "imagenet21k_ViT-S_16.npz",
    "sup_vits32_imagenet21k": "imagenet21k_ViT-S_32.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
}
class VptCfgNode():
    def __init__(self):
        super().__init__()
        self.NUM_TOKENS = 4
        self.DEEP = False
        self.DROPOUT = 0.0
        self.LOCATION = "prepend"
        self.INITIATION = "random"  # "final-cls", "cls-first12"
        self.CLSEMB_FOLDER = ""
        self.CLSEMB_PATH = ""
        self.PROJECT = -1  # "projection mlp hidden dim"
        self.NUM_DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
        self.REVERSE_DEEP = False  # if to only update last n layers, not the input layer
        self.DEEP_SHARED = False  # if true, all deep layers will be use the same prompt emb
        self.FORWARD_DEEP_NOEXPAND = False  # if true, will not expand input sequence for layers without prompt
        # how to get the output emb for cls head:
            # original: follow the orignial backbone choice,
            # img_pool: image patch pool only
            # prompt_pool: prompt embd pool only
            # imgprompt_pool: pool everything but the cls token
        self.VIT_POOL_TYPE = "original" 

def get_vpt_cfg(args):
    prompt_cfg = VptCfgNode()
    # prompt_cfg.DEEP = args.deep
    return prompt_cfg
    

def get_prompt_vit():
    pass
    #todo

class PromptViT(nn.Module):
    """ViT-related model."""

    def __init__(self, args, model_type="sup_vitb16_imagenet21k", vis=False):
        super().__init__()
        if args.mode.lower() in ['full', 'q-ffl', 'drfl']:
            prompt_cfg = None
        else:
            prompt_cfg = get_vpt_cfg(args)
        self.froze_enc = False
        self.model_type = model_type
        self.build_backbone(
            prompt_cfg, vis=vis)
        self.setup_side()
        self.setup_head(args.num_classes)
        print(args.num_classes)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]        

    def setup_side(self):
        self.side = None
        

    def build_backbone(self, prompt_cfg, vis):
        m2featdim = {
            "sup_vitb16_224": 768,
            "sup_vitb16": 768,
            "sup_vitl16_224": 1024,
            "sup_vitl16": 1024,
            "sup_vits16_imagenet21k": 384,
            "sup_vits32_imagenet21k": 384,
            "sup_vitb8_imagenet21k": 768,
            "sup_vitb16_imagenet21k": 768,
            "sup_vitb32_imagenet21k": 768,
            "sup_vitl16_imagenet21k": 1024,
            "sup_vitl32_imagenet21k": 1024,
            "sup_vith14_imagenet21k": 1280,
        }
        model_type = self.model_type
        crop_size=224
        model_root = "../pretrained/"
        self.feat_dim = m2featdim[model_type]
        if prompt_cfg is not None:
            self.enc = PromptedVisionTransformer(
                prompt_cfg, model_type,
                crop_size, num_classes=-1, vis=vis
            )
        else:
            print('type is full tune')
            self.enc = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis)

        print('loading pretrained weights...')
        self.enc.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))
        
    def setup_head(self, class_num):
        MLP_NUM = 0
        NUMBER_CLASSES = class_num
        # self.head = MLP(
        #     input_dim=self.feat_dim,
        #     mlp_dims=[self.feat_dim] * MLP_NUM + \
        #         [NUMBER_CLASSES], # noqa
        #     special_bias=True
        # )
        self.head = nn.Linear(self.feat_dim, NUMBER_CLASSES)

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x, x
        x = self.head(x)

        return x
        
    def forward_feature(self, feat):
        x = self.head(feat)
        return x

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa
                print('is deep prompt')
                total_d_layer = config.transformer["num_layers"]-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)


                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        print(img_size)
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]
        print(last_dim)

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x
