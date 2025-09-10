from transformers import AutoTokenizer
import torch 
import joblib 
import numpy as np
import pickle
import os
import re
import glob
import string
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import itertools
from utils import get_segmentation_indices,segmentation_to_word_list,model_to_path_dict

def generate_pairwise_events(tokenizer,tokens,event_start_indices,event_end_indices,pairwise_event_save_dir):
    bos_token_tensor = torch.Tensor([tokenizer.bos_token_id]).type(torch.int64) # int64 is longTensor
    num_events = len(event_start_indices)
    pairs = list(itertools.combinations(np.arange(num_events), 2))
    token_pairs_dict = {}
    startidx_of_second_event_dict = {} # 2nd chunk start token index without the bos token
    for i,pair in enumerate(pairs):
        event1 = pair[0]
        event2 = pair[1]
        event1_tokens = tokens[event_start_indices[event1]:event_end_indices[event1]]
        event2_tokens = tokens[event_start_indices[event2]:event_end_indices[event2]]
        token_pair = torch.cat((bos_token_tensor,event1_tokens,event2_tokens))
        token_pairs_dict[pair] = token_pair
        startidx_of_second_event_dict[pair] =len(event1_tokens)
        token_pair_wo_bos = token_pair[1:]
        assert torch.all(token_pair_wo_bos[len(event1_tokens):] ==event2_tokens)
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_tokens.pkl'),'wb') as f:
        pickle.dump(token_pairs_dict,f)
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_startidx_of_second_event.pkl'),'wb') as f:
        pickle.dump(startidx_of_second_event_dict,f)


def main(args):
    story = args.story
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'moth_stories_output')
    # load consensus
    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%args.story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    # with open(os.path.join(args.segmentation_dir,'%s_consensus_wordlist.pkl'%args.story),'rb') as f:
    #     consensus_wordlist = pickle.load(f)
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%args.story),'r') as f:
        original_txt = f.read()
    # load story tokens
    story_tokens = torch.load(os.path.join(moth_output_dir,story,'tokens.pkl'))
    with open(os.path.join(moth_output_dir,story,'tokenized_txt.pkl'),'rb') as f:
        tokenized_txt = pickle.load(f)
    segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char = model_to_path_dict[args.model]['initial_char'])
    event_end_indices = np.array(segmentation_indices_in_tokens)+1 # [start:end] gives you the event
    event_start_indices = np.insert(event_end_indices[:-1],0,0)
    story_tokens = story_tokens[0,1:] # get rid of bos token
    if story_tokens.get_device()>=0:
        story_tokens = story_tokens.detach().cpu()
    pairwise_event_save_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event',story)

    if args.split_story_by_duration:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_adjusted')
            story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df_adjusted.csv'))
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
            story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'))
        split_event_start_indices = story_split_by_duration_df['event_starts']
        split_event_end_indices = story_split_by_duration_df['event_ends']
    elif args.split_story_by_tokens:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_adjusted')
            story_split_by_token_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df_adjusted.csv'))
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens')
            story_split_by_token_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df.csv'))
        split_event_start_indices = story_split_by_token_df['event_starts']
        split_event_end_indices = story_split_by_token_df['event_ends']


    if not os.path.exists(pairwise_event_save_dir):
        os.makedirs(pairwise_event_save_dir)
    if args.split_story_by_duration or args.split_story_by_tokens:
        generate_pairwise_events(tokenizer,story_tokens,split_event_start_indices,split_event_end_indices,pairwise_event_save_dir)
    else: # no recombination
        generate_pairwise_events(tokenizer,story_tokens,event_start_indices,event_end_indices,pairwise_event_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--segmentation_dir",default = '/home/jianing/generation/behavior_data/segmentation')
    parser.add_argument("--moth_output_dir",default = '/home/jianing/generation/generated/')
    parser.add_argument("--save_dir",default='/home/jianing/generation/generated/')
    parser.add_argument("--original_transcript_dir",default = "/home/jianing/generation/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--unk", action = 'store_true',help = "whether to insert an unk token at point of ablation")
    parser.add_argument("--story",default = 'pieman')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--adjusted",action = 'store_true',help = 'use manually adjusted boundaries that respect phrase boundaries')
    parser.add_argument("--split_story_by_duration", action = 'store_true',help = "divide entire story into equal duration chunks")
    parser.add_argument("--split_story_by_tokens", action = 'store_true',help = "divide entire story into equal #token chunks")
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks for even split')
    args = parser.parse_args()
    main(args)

