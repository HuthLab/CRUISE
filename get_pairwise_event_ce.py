from transformers import AutoTokenizer,AutoModelForCausalLM
import torch 
import numpy as np
import pickle
import os
from torch.nn import functional as F
from natsort import os_sorted
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
from utils import calculate_cross_entropy,model_to_path_dict
import gc

def main(args):
    story = args.story
    pairwise_event_save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event',story)

    if args.split_story_by_duration:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_adjusted')
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
    elif args.split_story_by_tokens:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_adjusted')
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens')

    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_tokens.pkl'),'rb') as f:
        token_pairs_dict = pickle.load(f)
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    model = AutoModelForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],device_map='auto',torch_dtype = torch.float16)
    pairwise_event_ce = {}
    for pair in tqdm(token_pairs_dict.keys()):
        input_token = token_pairs_dict[pair]
        with torch.no_grad():
            output = model(torch.unsqueeze(input_token,0).to(device),return_dict = True)
        logits = output['logits'].detach().cpu()
        entropy = calculate_cross_entropy(input_token[1:],logits[0,:-1],base2=True) # excluding bos token 
        pairwise_event_ce[pair] = entropy
        del output, logits
        gc.collect()
        torch.cuda.empty_cache()
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_ce.pkl'),'wb') as f:
        pickle.dump(pairwise_event_ce,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir",default='/work/09192/jianing/ls6/Memory_generation/generated')
    parser.add_argument("--story",default = 'pieman')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--adjusted",action = 'store_true',help = 'use manually adjusted boundaries that respect phrase boundaries')
    parser.add_argument("--split_story_by_duration", action = 'store_true',help = "divide entire story into equal duration chunks")
    parser.add_argument("--split_story_by_tokens", action = 'store_true',help = "divide entire story into equal #token chunks")
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks for even split')
    args = parser.parse_args()
    main(args)