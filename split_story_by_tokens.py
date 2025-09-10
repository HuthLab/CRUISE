from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import numpy as np
import pickle
import glob
import re
import matplotlib.pyplot as plt
import itertools
from utils import get_segmentation_indices,segmentation_to_word_list
from utils import model_to_path_dict
from split_event_by_timing import align_timing_with_text
import argparse

def check_is_boundary(split_df_timing,segmentation_indices_in_tokens,story):
    num_segmentations = []
    # determine if there's a boundary in the chunk
    for i,row in split_df_timing.iterrows():
        chunk_start = row['event_starts']
        chunk_end = row['event_ends']
        segmentation_in_chunk = [idx for idx in segmentation_indices_in_tokens[:-1] if idx>=chunk_start and idx < chunk_end]
        num_segmentations.append(len(segmentation_in_chunk))
    num_segmentations = np.array(num_segmentations)
    split_df_timing['num_boundaries'] = num_segmentations
    split_df_timing['has_boundary'] = num_segmentations>0
    print(story,'boundary:',np.sum(num_segmentations>0),'non-boundary:',np.sum(num_segmentations==0),'num events:',len(segmentation_indices_in_tokens))
    return split_df_timing

def get_split_by_token_df(factor,consensus_txt,story_tokens,tokenized_txt,timing_df,original_txt,story,segmentation_indices_in_tokens,tokenizer,model_initial_char):
    num_events = len(consensus_txt)
    story_token_len = story_tokens.shape[0]
    spacing = np.linspace(0,story_token_len,int(num_events*factor)+1).astype(int)
    token_start_indices = spacing[:-1]
    token_end_indices = spacing[1:]
    chunk_token_len = token_end_indices-token_start_indices
    
    # make sure it doesn't split within a word
    for i in range(len(token_start_indices)):
        if i==0:
            continue
        start_idx = token_start_indices[i]
        while model_initial_char not in tokenized_txt[start_idx]:
            print(tokenized_txt[start_idx])
            token_start_indices[i]+=1
            token_end_indices[i-1]+=1
            start_idx = token_start_indices[i]
    chunk_token_len = token_end_indices-token_start_indices
    assert np.all(token_start_indices[1:]==token_end_indices[:-1])
    
    # get the text of each chunk
    chunk_txt = []
    for start_idx,end_idx in zip(token_start_indices,token_end_indices):
        this_chunk_txt = tokenizer.decode(story_tokens[start_idx:end_idx])
        if this_chunk_txt not in original_txt:
            this_chunk_txt = ''.join([s.replace(model_initial_char,' ') for s in tokenized_txt[start_idx:end_idx]]) 
            if this_chunk_txt not in original_txt:
                print(this_chunk_txt)
        assert this_chunk_txt in original_txt
        chunk_txt.append(this_chunk_txt)
    split_df = pd.DataFrame({'text':chunk_txt,
                        'event_starts':token_start_indices,
                        'event_ends':token_end_indices,
                        'event_len':chunk_token_len})
    # get the onset, offset and duration of each chunk
    split_df_timing = align_timing_with_text(split_df,timing_df,original_txt,story)
    split_df_timing = check_is_boundary(split_df_timing,segmentation_indices_in_tokens,story)
    return split_df_timing

def main(args):
    story = args.story
    if args.model =='Llama3.2-3b-instruct_finetuned':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    
    model_initial_char = model_to_path_dict[args.model]['initial_char']
    model_save_dir_name = model_to_path_dict[args.model]['save_dir_name']
    moth_output_dir = os.path.join(args.save_dir,model_save_dir_name,'moth_stories_output')
    # tokenized txt 
    with open(os.path.join(moth_output_dir,story,'tokenized_txt.pkl'),'rb') as f:
        tokenized_txt = pickle.load(f)
    story_tokens = torch.load(os.path.join(moth_output_dir,story,'tokens.pkl'))
    story_tokens = story_tokens[0,1:].cpu().detach()
    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
        original_txt = f.read()

    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char='Ġ')
    event_len = np.diff([0]+segmentation_indices_in_tokens)
    event_len[0]+=1

    timing_df = pd.read_csv(os.path.join(args.timing_dir,'%s_timing.csv'%story))
    # for souls, the timing_df is missing two empty strings. add them in to match the consensus wordlist
    if len(timing_df) != len(consensus_wordlist):
        empty_string_idx = []
        for i in range(len(consensus_wordlist)):
            if consensus_wordlist[i] =='':
                print(i)
                empty_string_idx.append(i)
        for idx in empty_string_idx:
            row = pd.DataFrame({"text": '', "start": np.nan,"stop":np.nan}, index=[idx])
            timing_df = pd.concat([timing_df.iloc[:idx], row, timing_df.iloc[idx:]]).reset_index(drop=True)
    if story =='souls':
        timing_df['text'] = [t[:-1] if t!= '' and t[-1] == ' ' else t for t in timing_df['text']]
    assert len(timing_df) == len(consensus_wordlist)

    event_timing_dir = os.path.join(args.event_timing_dir,story)
    if not os.path.exists(event_timing_dir):
        os.makedirs(event_timing_dir)
    split_df_timing = get_split_by_token_df(args.factor,consensus_txt,story_tokens,tokenized_txt,timing_df,original_txt,story,segmentation_indices_in_tokens,tokenizer,model_initial_char)
    split_df_timing['story'] = story
    split_df_timing.to_csv(os.path.join(event_timing_dir,'story_even_token_factor_%.1f.csv'%args.factor),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--story")
    parser.add_argument("--save_dir",default = '../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--timing_dir",default = "../behavior_data/transcripts/timing",help = "directory storing timing dfs")
    parser.add_argument("--event_timing_dir", default = '../behavior_data/story_split_timing')
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks')
    args = parser.parse_args()
    main(args)