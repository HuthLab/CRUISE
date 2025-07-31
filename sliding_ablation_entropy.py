# inference to get ablation entropy for sliding window entropy 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig,LogitsProcessorList,MinLengthLogitsProcessor,ConstrainedBeamSearchScorer,PhrasalConstraint
import torch 
import numpy as np
import pandas as pd
import pickle
import accelerate
import os
import re 
import glob
from tqdm.notebook import tqdm
from torch.nn import functional as F
import argparse
from utils import model_to_path_dict,calculate_cross_entropy


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    model = AutoModelForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],device_map='auto',torch_dtype = torch.float16)
    device = 'cuda'

    ### sliding window ablation 
    if args.model=='Llama-2-13b-chat-hf':
        moth_ablation_dir = '/work/09192/jianing/ls6/Memory_generation/ablation/sliding_window_ablation/moth_stories'
    elif args.model=='Llama3-8b-instruct':
        moth_ablation_dir = os.path.join(args.ablation_dir,model_to_path_dict[args.model]['save_dir_name'],'sliding_window_ablation/moth_stories')
    device = 'cuda'
    
    stories = os.listdir(moth_ablation_dir)
    for story in tqdm(stories):
        story_stim_dir = os.path.join(moth_ablation_dir,story)
        if os.path.exists(os.path.join(story_stim_dir,'ablation_logits','ablation_cross_entropy_count_balanced.pkl')):
            print('skipping %s, already exists'%story)
            continue
        with open(os.path.join(story_stim_dir,'ablation_stim_count_balanced.pkl'),'rb') as f:
            random_ablation_stim=pickle.load(f)
        token_tensors = random_ablation_stim['post_ablation_tokens']
        # inference 
        all_entropy = []
        for input_token in tqdm(token_tensors):
            with torch.no_grad():
                output = model(torch.unsqueeze(input_token,0).to(device),return_dict = True)
            logits = output['logits'].detach().cpu()
            entropy = calculate_cross_entropy(input_token[1:],logits[0,:-1],base2=True) # excluding bos token 
            all_entropy.append(entropy)
        if not os.path.exists(os.path.join(story_stim_dir,'ablation_logits')):
            os.makedirs(os.path.join(story_stim_dir,'ablation_logits'))
        with open(os.path.join(story_stim_dir,'ablation_logits','ablation_cross_entropy_count_balanced.pkl'),'wb') as f:
            pickle.dump(all_entropy,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--ablation_dir",default = '/work/09192/jianing/ls6/Memory_generation/ablation')
    args = parser.parse_args()
    main(args)