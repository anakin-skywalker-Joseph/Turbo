import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import minigpt4.turbo as turbo
import time
from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config
import torch.backends.cudnn as cudnn
import random
import ipdb
    # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
    # answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
from fvcore.nn import FlopCountAnalysis
from torch import nn

conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""

def print_params_and_flops(model, device, config=None):
    model.eval()
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, inputs):
            # question = "What is inside the picture?"
            # text = prepare_texts(question, conv_temp)
            return self.model.visual_encoder(inputs)
    with torch.no_grad():
        wrapper_model = Wrapper(model); 
        inputs = torch.randn(1, 3, 448, 448).to(device)
        flop = FlopCountAnalysis(wrapper_model, inputs)
        #print(flop_count_table(flop, max_depth=7, show_param_shapes=True))
        print("Total", flop.total() / 1e9)
        return flop.total() / 1e9


@torch.no_grad()
def main(args, model, r_value=12, alpha=1):   
    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = 12
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating model")
    if r_value!=0:  
        if alpha!=0:
            turbo.patch.turbo(model.visual_encoder)
            model.visual_encoder.r=r_value
            model.visual_encoder.alpha=alpha
        else:
            turbo.patch.turbo_tome(model.visual_encoder)
            model.visual_encoder.r=r_value
    is_cuda = True 
    input_size=(3,448,448)
    model = model.eval().to(device)   
    batch_size = 10
    runs=70
    warm_up=9
    total=0
    flops=print_params_and_flops(model,device)
    input = torch.rand(batch_size, *input_size, device=device)   
    question = "What is in the picture?"
    texts = prepare_texts(question, conv_temp)  # warp the texts with conversation template
    with torch.autocast(device.type):
        with torch.no_grad():   
            for i in tqdm(range(runs), desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()
                answers = model.generate(input, texts, max_new_tokens=5, do_sample=False)
                total += batch_size  
    # if is_cuda:
    #     torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start   

    throughput = total / elapsed   ## 吞吐

    print(f"r: {r_value}, Throughput: {throughput:.2f} im/s")
    with open("accelerate_vqa.txt","a") as ac:
        ac.write(f"r: {r_value}, Throughput: {throughput:.2f} im/s\n")
    flops=print_params_and_flops(model,device)
    with open("accelerate_vqa.txt","a") as ac:
        ac.write(f"r: {r_value}, flops:{flops}\n") 
    print(f"r: {r_value}, Flops: {flops:.2f}")
    return throughput                 
    
            

if __name__ == '__main__':
    def list_of_str(arg):
        return list(map(str, arg.split(',')))

    parser = eval_parser()
    args = parser.parse_args()
    cfg = Config(args)

    model, vis_processor = init_model(args)
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    model.eval()
    #r_val=[5*i for i in range(15)]
    r_val=[0,8,12,16]
    # alpha_val=[0,1]
    for r in r_val:
        # for a in alpha_val:
        main(args, model, r, 1)
        time.sleep(2)