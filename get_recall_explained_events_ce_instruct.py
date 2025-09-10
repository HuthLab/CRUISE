from transformers import AutoTokenizer,AutoModelForCausalLM
import torch 
import numpy as np
import pickle
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
from utils import calculate_cross_entropy,model_to_path_dict
import gc

def run_inference(input_tokens_list,model,device):
    ce = []
    for i,input_tokenized in enumerate(input_tokens_list):
        with torch.no_grad():
            output = model(input_tokenized.to(device),return_dict = True)
            logits = output['logits']
        # calculate cross entropy
        entropy = calculate_cross_entropy(input_tokenized[0,1:],logits[0,:-1],base2=True) # excluding bos token 
        ce.append(entropy)
        del output, logits
        gc.collect()
        torch.cuda.empty_cache()
    return ce

def run_inference_nested(stim_dict,model,device):
    nested_ce_dict = {}
    for event in stim_dict.keys():
        input_tokens_list = stim_dict[event]['input_tokens']
        ce = run_inference(input_tokens_list,model,device)
        nested_ce_dict[event] = ce
    return nested_ce_dict

def main(args):
    story = args.story
    model = AutoModelForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],device_map='auto',torch_dtype = torch.float16)
    device = 'cuda'
    pairwise_event_save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event',story)

    if args.split_story_by_duration:
        if args.adjusted:
            if args.factor is not None:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_adjusted')
        else:
            if args.factor is not None:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
    elif args.split_story_by_tokens:
        if args.adjusted:
            if args.factor is not None:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
            else:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_adjusted')
        else:
            if args.factor is not None:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f'%args.factor)
            else:
                save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens')
    else:
        save_dir = pairwise_event_save_dir

    if args.random_recalls:
        save_dir = os.path.join(save_dir,'random_recalls')
    save_dir = os.path.join(save_dir,'instruct')

    if args.recall_event_concat:
        with open(os.path.join(save_dir,'event_only_stim_instruct.pkl'),'rb') as f:
            event_only_stim_dict = pickle.load(f)
        event_only_ce = run_inference(event_only_stim_dict['input_tokens'],model,device)
        with open(os.path.join(save_dir,'event_only_ce_instruct.pkl'),'wb') as f:
            pickle.dump(event_only_ce,f)
        del event_only_ce

        with open(os.path.join(save_dir,'recall_event_stim_instruct.pkl'),'rb') as f:
            recall_event_stim_dict = pickle.load(f)
        recall_event_ce_dict = run_inference_nested(recall_event_stim_dict,model,device)
        with open(os.path.join(save_dir,'recall_event_ce_instruct.pkl'),'wb') as f:
            pickle.dump(recall_event_ce_dict,f)
        del recall_event_ce_dict
    if args.event_recall_concat:
        with open(os.path.join(save_dir,'recall_only_stim_instruct.pkl'),'rb') as f:
            recall_instruct_stim_dict = pickle.load(f)
        recall_only_ce = run_inference(recall_instruct_stim_dict['input_tokens'],model,device)
        with open(os.path.join(save_dir,'recall_only_ce_instruct.pkl'),'wb') as f:
            pickle.dump(recall_only_ce,f)
        del recall_only_ce

        with open(os.path.join(save_dir,'event_recall_stim_instruct.pkl'),'rb') as f:
            event_recall_stim_dict = pickle.load(f)
        event_recall_ce_dict = run_inference_nested(event_recall_stim_dict,model,device)
        with open(os.path.join(save_dir,'event_recall_ce_instruct.pkl'),'wb') as f:
            pickle.dump(event_recall_ce_dict,f)
        del event_recall_ce_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir",default = '../generated')
    parser.add_argument("--story",default = 'pieman',help = 'to run the concatenated entropy of original stories, enter original')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--adjusted",action = 'store_true',help = 'use manually adjusted boundaries that respect phrase boundaries')
    parser.add_argument("--recall_event_concat",action = 'store_true',help = 'parse recall first, event next concatentation')
    parser.add_argument("--event_recall_concat",action = 'store_true',help = 'parse event first, recall next concatenation')
    parser.add_argument("--split_story_by_duration", action = 'store_true',help = "divide entire story into equal duration chunks")
    parser.add_argument("--split_story_by_tokens", action = 'store_true',help = "divide entire story into equal #token chunks")
    parser.add_argument("--random_recalls", action = 'store_true',help = "use recalls randomly sampled from the other stories")
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks for even split')
    args = parser.parse_args()
    main(args)