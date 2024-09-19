from typing import Tuple
import torch
#from timm.models.vision_transformer import Attention, Block, VisionTransformer
import sys
# sys.path.append("../")
# sys.path.append("../../")
# sys.path.append("../../models")
from minigpt4.models.eva_vit import Attention, Block, VisionTransformer
from minigpt4.turbo.merge import bipartite_soft_matching, merge_wavg, bipartite_unimodal_matching,merge_wavg_ours
from minigpt4.turbo.utils import parse_r
import ipdb
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from minigpt4.models.eva_vit import Mlp
import torch.nn.functional as F

class TurboBlock_raw(Block):                                          ## 整个模块中插入 Turbo Inference 模块
    # def _drop_path1(self, x):   
    #     return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    # def _drop_path2(self, x):
    #     return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, rel_pos_bias=None):   ## x.shape=[256,197,768]
        # attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        # x_attn, metric = self.attn(self.norm1(x), attn_size)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]
        # x = x + x_attn                          ## x.shape = [256, 197, 768]
        # r = self._turbo_info["r"].pop(0)

        # ## 根据输入的r决定要不要 合并token做加速
        # if r > 0:
        #     merge = bipartite_soft_matching(metric, r, self._turbo_info["class_token"],self._turbo_info["r_threshold"])
        #     x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])  ## token合并
            
        # x = x + self.mlp(self.norm2(x))         ## x.shape = [256,197-r=133,768]
        # return x
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None

        if self.gamma_1 is None:
            x_attn, metric = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(x_attn)
            r = self._turbo_info["r"].pop(0)

            if r > 0:  ## 根据输入的r决定要不要 合并token做加速
                merge = bipartite_soft_matching(metric, r, self._turbo_info["class_token"])
                x, self._turbo_info["size"] = merge_wavg_ours(merge, x, self._turbo_info["size"])  ## token合并

            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, metric = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(self.gamma_1 * x_attn)
            r = self._turbo_info["r"].pop(0)

            if r > 0:  ## 根据输入的r决定要不要 合并token做加速
                merge = bipartite_soft_matching(metric, r, self._turbo_info["class_token"],self._turbo_info["r_threshold"])
                x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])  ## token合并
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TurboAttention_raw(Attention): 
    def forward(self, x: torch.Tensor, size: torch.Tensor = None, rel_pos_bias=None):
        # B, N, C = x.shape  
        # qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        # q, k, v = (qkv[0], qkv[1], qkv[2])  
        # attn = (q @ k.transpose(-2, -1)) * self.scale 
          
        # if size is not None:
        #     attn = attn + size.log()[:, None, None, :, 0]

        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)  
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
        # x = self.proj(x)                                
        # x = self.proj_drop(x)                            

        # return x, k.mean(1)
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)                           

        return x, k.mean(1)

def make_turbo_class_raw(transformer_class):
    class TurboVisionTransformer_raw(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info["r"] = parse_r(len(self.blocks), self.r)
            self._turbo_info["size"] = None
            self._turbo_info["source"] = None
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

    return TurboVisionTransformer_raw

class TurboAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor = None, rel_pos_bias=None):
        # B, N, C = x.shape  
        # qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        # q, k, v = (qkv[0], qkv[1], qkv[2])  
        # attn = (q @ k.transpose(-2, -1)) * self.scale 
        # if size is not None:
        #     attn = attn + size.log()[:, None, None, :, 0]

        # attn = attn.softmax(dim=-1)  #(B, num_heads, N_token, N_token)
        # attn_cls=torch.mean(attn[...,0,:],dim=1)  #(B, N_token)
        # attn = self.attn_drop(attn)  
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
        # x = self.proj(x)                                
        # x = self.proj_drop(x)                            
        # return x, k.mean(1),attn_cls
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn_cls = torch.mean(attn[...,0,:],dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)                           

        return x, k.mean(1), attn_cls

class TurboBlock(Block):                                          ## 整个模块中插入 Turbo Inference 模块
    # def _drop_path1(self, x):   
    #     return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    # def _drop_path2(self, x):
    #     return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    def forward(self, x: torch.Tensor, rel_pos_bias=None):   ## x.shape=[256,197,768]
        # attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        # x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]                     
        # x = x + x_attn                                            ## x.shape = [256, 197, 768]
        # r = self._turbo_info["r"].pop(0)
        # if r > 0:  ## 根据输入的r决定要不要 合并token做加速
        #     merge = bipartite_unimodal_matching(metric, 
        #                                         attn_cls,
        #                                         r,
        #                                         self._turbo_info["class_token"], 
        #                                         self._turbo_info["alpha"],
        #                                         self._turbo_info["num_layer"],
        #                                         self._turbo_info["beta"],
        #                                         self._turbo_info["gamma"],
        #                                         self._turbo_info["r_threshold"])
        #     x, self._turbo_info["size"] = merge_wavg_ours(merge, x, self._turbo_info["size"])  ## token合并           
        #     self._turbo_info["num_layer"]+=1           
        # x = x + self.mlp(self.norm2(x))                                    ## x.shape = [256,197-r=181,768]
        # return x
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None

        if self.gamma_1 is None:
            x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(x_attn)
            r = self._turbo_info["r"].pop(0)

            if r > 0:  ## 根据输入的r决定要不要 合并token做加速
                merge = bipartite_unimodal_matching(metric, 
                                                attn_cls,
                                                r,
                                                self._turbo_info["class_token"], 
                                                self._turbo_info["alpha"],
                                                self._turbo_info["num_layer"],
                                                self._turbo_info["beta"],
                                                self._turbo_info["gamma"],
                                                self._turbo_info["r_threshold"])
                x, self._turbo_info["size"] = merge_wavg_ours(merge, x, self._turbo_info["size"])  ## token合并

            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(self.gamma_1 * x_attn)
            r = self._turbo_info["r"].pop(0)

            if r > 0:  ## 根据输入的r决定要不要 合并token做加速
                merge = bipartite_unimodal_matching(metric,                                                 
                                                attn_cls,
                                                r,
                                                self._turbo_info["class_token"], 
                                                self._turbo_info["alpha"],
                                                self._turbo_info["num_layer"],
                                                self._turbo_info["beta"],
                                                self._turbo_info["gamma"],
                                                self._turbo_info["r_threshold"])
                x, self._turbo_info["size"] = merge_wavg_ours(merge, x, self._turbo_info["size"])  ## token合并
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def make_turbo_class(transformer_class):
    class TurboVisionTransformer(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info["r"] = parse_r(len(self.blocks), self.r)
            self._turbo_info["size"] = None
            self._turbo_info["source"] = None
            self._turbo_info["alpha"] = self.alpha
            self._turbo_info["beta"] = self.beta
            self._turbo_info["gamma"] =self.gamma
            self._turbo_info["num_layer"]=self.num_layer
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

        def generate(self, samples,*args, **kwdargs):
            self._turbo_info_visual["r"] = parse_r(len(self.visual_encoder.blocks), self.r)
            self._turbo_info_visual["size"] = None
            self._turbo_info_visual["source"] = None
            self._turbo_info["alpha"] = self.alpha
            self._turbo_info["beta"] = self.beta
            self._turbo_info["gamma"] =self.gamma
            self._turbo_info["num_layer"]=self.num_layer
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().generate(samples,*args, **kwdargs)

        def forward_image(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info_visual["r"] = parse_r(len(self.visual_encoder.blocks), self.r)
            self._turbo_info_visual["size"] = None
            self._turbo_info_visual["source"] = None
            self._turbo_info["alpha"] = self.alpha
            self._turbo_info["beta"] = self.beta
            self._turbo_info["gamma"] =self.gamma
            self._turbo_info["num_layer"]=self.num_layer
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward_image(*args, **kwdargs)

        def forward_text(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info_visual["r"] = parse_r(len(self.visual_encoder.blocks), self.r)
            self._turbo_info_visual["size"] = None
            self._turbo_info_visual["source"] = None
            self._turbo_info["alpha"] = self.alpha
            self._turbo_info["beta"] = self.beta
            self._turbo_info["gamma"] =self.gamma
            self._turbo_info["num_layer"]=self.num_layer
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward_text(*args, **kwdargs)
    return TurboVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    TurboVisionTransformer = make_turbo_class(model.__class__)    
    model.__class__ = TurboVisionTransformer
    model.r = 0 
    model.alpha=1
    model.beta=1   #控制衰减底数
    model.gamma=0  #控制衰减指数
    model.num_layer=0
    model.r_threshold = 40
    model._turbo_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None, "alpha":model.alpha,
        "beta":model.beta,"gamma":model.gamma,"num_layer":0,"r_threshold" :model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TurboBlock
            module._turbo_info = model._turbo_info
        elif isinstance(module, Attention):
            module.__class__ = TurboAttention

    # for module in model.modules():
    #     if isinstance(module, Attention) or isinstance(module, TurboAttention):
    #         module.__class__ = TurboAttention
    #     elif isinstance(module, Block) or isinstance(module, TurboBlock):
    #         module.__class__ = TurboBlock
    #         module._turbo_info = model._turbo_info

def apply_patch_tome(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    TurboVisionTransformer_raw = make_turbo_class_raw(model.__class__)    
    model.__class__ = TurboVisionTransformer_raw
    model.r = 0 
    model.alpha=1
    model.r_threshold = 40
    model._turbo_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None,"alpha":model.alpha,"r_threshold":model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TurboBlock_raw
            module._turbo_info = model._turbo_info
        elif isinstance(module, Attention):
            module.__class__ = TurboAttention_raw




