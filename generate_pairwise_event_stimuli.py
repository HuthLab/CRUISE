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
from utils import get_segmentation_indices,segmentation_to_word_list,model_to_path_dict,get_sherlock_event_indices

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

def get_recombined_boundaries(segmentation_indices_in_tokens):
    event_len = np.diff([0]+segmentation_indices_in_tokens)
    event_len[0]+=1
    event_end_indices = np.array(segmentation_indices_in_tokens)+1 # [start:end] gives you the event
    event_start_indices = np.insert(event_end_indices[:-1],0,0)

    partial_event_starts = [] # [start:end] gives you the event
    partial_event_ends = []
    event_type = []
    for event in range(len(event_len)):
        this_event_len = event_len[event]
        quarter_event_len = np.round(this_event_len/4)
        this_event_start = event_start_indices[event]
        partial_event_start = this_event_start
        this_event_end = event_end_indices[event]
        for i in range(4):
            partial_event_starts.append(partial_event_start)
            if not i==3:
                partial_event_end = partial_event_start+quarter_event_len
            else:
                partial_event_end = this_event_end
            partial_event_ends.append(partial_event_end)
            partial_event_start = partial_event_end
            if i ==0 or i ==3:
                event_type.append('boundary')
            else:
                event_type.append('inner')
    partial_event_starts = np.array(partial_event_starts).astype(int)
    partial_event_ends = np.array(partial_event_ends).astype(int)

    recombined_event_starts = []
    recombined_event_ends = []
    recombined_event_type = []
    for event in range(len(event_type)):
        partial_event_start = partial_event_starts[event]
        partial_event_end = partial_event_ends[event]
        if event ==0 or event == len(event_type)-1:
            recombined_event_starts.append(partial_event_start)
            recombined_event_ends.append(partial_event_end)
            recombined_event_type.append(event_type[event])
        else:
            if event_type[event+1] == event_type[event]:
                recombined_event_starts.append(partial_event_start)
                recombined_event_ends.append(partial_event_ends[event+1])
                recombined_event_type.append(event_type[event])
            else:
                continue
    return recombined_event_starts,recombined_event_ends,recombined_event_type

def main(args):
    story = args.story
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    if args.story=='sherlock':
        if args.twosessions:
            moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_2sessions')
            with open(os.path.join(args.sherlock_transcript_dir,'transcript_for_recall_2sessions.txt'),'r') as f:
                story_transcript = f.read()
            sherlock_event_df = pd.read_csv(os.path.join(args.sherlock_transcript_dir,'chen_scene_timing_sec_lines_noopening.csv'))
        
        else:
            if args.recombine and not args.adjusted: 
                moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock')
                sherlock_event_df = pd.read_csv(os.path.join(args.sherlock_transcript_dir,'truncated_scene_timing_sec.csv'))
            else:
                moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_truncated')
                sherlock_event_df = pd.read_csv(os.path.join(args.sherlock_transcript_dir,'truncated_scene_timing_sec.csv'))
        # load story tokens
        story_tokens = torch.load(os.path.join(moth_output_dir,'tokens.pkl'))
        story_tokens = story_tokens[0,1:]
        with open(os.path.join(moth_output_dir,'tokenized_txt.pkl'),'rb') as f:
            tokenized_txt = pickle.load(f)
        sherlock_event_df = get_sherlock_event_indices(sherlock_event_df,tokenizer,story_tokens,story_transcript,args.model)
        event_end_indices = sherlock_event_df['end_idx_in_tokens'] # [start:end] gives you the event
        event_start_indices = sherlock_event_df['start_idx_in_tokens']
    else:
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

    if args.story =='sherlock' and args.twosessions:
        pairwise_event_save_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event','sherlock_2sessions')
    else:
        pairwise_event_save_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event',story)

    if args.recombine:
        if args.adjusted: # the adjusted ones all have event starts and ends in tokens calculated
            pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'recombine_adjusted')
            recombined_event_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'adjusted_recombined_event_df.csv'))
            recombined_event_starts = recombined_event_df['recombined_event_starts']
            recombined_event_ends = recombined_event_df['recombined_event_ends']
        else:
            pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'recombine')
            if args.story =='sherlock': # sherlock is actually divided by time, and the boundary indices are already calculated
                recombined_event_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'recombined_event_df.csv'))
                recombined_event_starts = recombined_event_df['recombined_event_starts']
                recombined_event_ends = recombined_event_df['recombined_event_ends']
            else:
                recombined_event_starts,recombined_event_ends,recombined_event_type = get_recombined_boundaries(segmentation_indices_in_tokens)
    elif args.recombine_duration:
        if args.adjusted:
            pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'recombine_duration_adjusted')
            recombined_event_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'adjusted_recombined_event_df.csv'))
        else:
            pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'recombine_duration')
            recombined_event_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'recombined_event_df.csv'))
        recombined_event_starts = recombined_event_df['recombined_event_starts']
        recombined_event_ends = recombined_event_df['recombined_event_ends']
    elif args.split_story_by_duration:
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
    if args.recombine:
        generate_pairwise_events(tokenizer,story_tokens,recombined_event_starts,recombined_event_ends,pairwise_event_save_dir)
        if not args.adjusted and args.story!='sherlock': # those have already made the df 
            recombined_event_df = pd.DataFrame({'recombined_event_starts':recombined_event_starts,
                                                'recombined_event_ends':recombined_event_ends,
                                                'recombined_event_type':recombined_event_type})
            recombined_event_df.to_csv(os.path.join(pairwise_event_save_dir,'recombined_event_df.csv'),index = False)
    elif args.recombine_duration:
        generate_pairwise_events(tokenizer,story_tokens,recombined_event_starts,recombined_event_ends,pairwise_event_save_dir)
    elif args.split_story_by_duration or args.split_story_by_tokens:
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
    parser.add_argument("--recombine", action = 'store_true',help = "divide each event into 4 parts and recombine with neighboring events")
    parser.add_argument("--adjusted",action = 'store_true',help = 'use manually adjusted boundaries that respect phrase boundaries')
    parser.add_argument("--recombine_duration", action = 'store_true',help = "divide events by time instead of tokens and recombine them")
    parser.add_argument("--sherlock_transcript_dir",default = '/home/jianing/generation/sherlock')
    parser.add_argument("--twosessions",action = 'store_true',help = 'use recall and transcripts from both sessions of sherlock')
    parser.add_argument("--split_story_by_duration", action = 'store_true',help = "divide entire story into equal duration chunks")
    parser.add_argument("--split_story_by_tokens", action = 'store_true',help = "divide entire story into equal #token chunks")
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks for even split')
    args = parser.parse_args()
    main(args)

