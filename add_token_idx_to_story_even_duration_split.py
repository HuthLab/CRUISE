from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import get_segmentation_indices,segmentation_to_word_list
from utils import calculate_cross_entropy,normalize_entropy,model_to_path_dict
from split_event_by_timing import align_timing_with_text
from split_story_by_tokens import check_is_boundary
import argparse
import string
from tqdm import tqdm

def find_sublist(big_list, sublist,last_end_idx):
    sublist_length = len(sublist)
    for i in range(len(big_list) - sublist_length + 1):
        if i <last_end_idx:
            continue
        if big_list[i:i + sublist_length] == sublist:
            return i, i + sublist_length  # Return start and end indices, end index is exclusive (ie [start,end] gives this sublist)
    return -1, -1  # Return (-1, -1) if the sublist is not found

def find_subtensor_indices(a, b):
    # Create a sliding window to find where tensor a occurs in tensor b
    for i in range(b.size(0) - a.size(0) + 1):
        if torch.equal(b[i:i + a.size(0)], a):
            start_index = i
            end_index = i + a.size(0)
            return start_index, end_index
    return None  # If a is not a sub-tensor of b

def align_to_token_indices(manual_adjusted_recombined_event_df,original_txt,tokenized_txt,story_tokens,tokenizer,model_name):
    manual_adjusted_event_txt = manual_adjusted_recombined_event_df['text'].values
    adjusted_detail_tokens = []
    #[start_idx,end_idx], exclusive, (ie [start,end] gives this sublist)
    adjusted_detail_start_indices = []
    adjusted_detail_end_indices = []
    
    strip_chars_all = string.punctuation + '-–'
    strip_chars = strip_chars_all.translate(str.maketrans('', '', '\'')) # don't strip ' (quote)

    curr_start_idx = 0
    last_end_idx = 0
    for i,detail in enumerate(manual_adjusted_event_txt):
        detail = detail.translate(str.maketrans('', '', strip_chars)).lower()
        detail_split = detail.split()
        original_chunk_split = original_txt.split()[curr_start_idx:curr_start_idx+len(detail_split)]
        if detail_split != original_chunk_split:
            print(detail_split)
            print(original_chunk_split)
        assert detail_split == original_chunk_split
        if detail[-1] == ' ': # remove space of last word 
            detail = detail[:-1]
        
        if model_name =='gemma-2-9b-it': # gemma tokenizes double spaces separately as '__', then the next word without leading space
            detail_idx_in_original = original_txt.index(detail)
            if detail[0]==' ' and original_txt[detail_idx_in_original-1]==' ': # if detail starts with space and the previous char is also a space
                detail = detail[1:]
        if i!=0 and detail[0] != ' ': # add space before first word
            if model_name =='gemma-2-9b-it': # gemma tokenizes double spaces separately as '__', then the next word without leading space
                detail_idx_in_original = original_txt.index(detail)
                # so if there's a double space in front originally, don't add the space cuz the word will be tokenized separately anyway
                if original_txt[detail_idx_in_original-1]==' ' and original_txt[detail_idx_in_original-2]==' ':
                    pass
                else:
                    detail = ' '+ detail
            else:
                detail = ' '+ detail
        detail_tokenized_txt = tokenizer.tokenize(detail)
        start_idx, end_idx = find_sublist(tokenized_txt,detail_tokenized_txt,last_end_idx) # end index is exclusive (ie [start,end] gives this sublist)
        if start_idx==-1:
            print(detail)
            assert detail in original_txt
        assert start_idx !=-1,'detail is not found in big list'
        this_adjusted_detail_tokens = story_tokens[start_idx:end_idx]
        adjusted_detail_tokens.append(this_adjusted_detail_tokens)
        adjusted_detail_end_indices.append(end_idx)
        adjusted_detail_start_indices.append(start_idx)
        last_end_idx =end_idx
        curr_start_idx +=len(detail_split)
    manual_adjusted_recombined_event_df['event_starts'] = adjusted_detail_start_indices
    manual_adjusted_recombined_event_df['event_ends'] = adjusted_detail_end_indices
    return manual_adjusted_recombined_event_df

def main(args):
    story = args.story
    model_save_dir_name = model_to_path_dict[args.model]['save_dir_name']
    model_initial_char = model_to_path_dict[args.model]['initial_char']
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])

    moth_output_dir = os.path.join(args.save_dir,model_save_dir_name,'moth_stories_output',args.story)
    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
        original_txt = f.read()
            
    # tokenized txt 
    with open(os.path.join(moth_output_dir,'tokenized_txt.pkl'),'rb') as f:
        tokenized_txt = pickle.load(f)
    story_tokens = torch.load(os.path.join(moth_output_dir,'tokens.pkl'))
    story_tokens = story_tokens[0,1:].cpu().detach()
    
    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char=model_initial_char)

    event_timing_dir = os.path.join(args.event_timing_dir,story)
    
    if args.even_tokens and args.parse_adjusted: # split by tokens
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

        adjusted_equal_token_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_token_factor_%.1f_adjusted.csv'%args.factor))
        # align to tokens
        adjusted_equal_token_df_tokens = align_to_token_indices(adjusted_equal_token_df_text,original_txt,tokenized_txt,story_tokens,tokenizer,args.model)
        # align to timing
        adjusted_equal_token_df_tokens = align_timing_with_text(adjusted_equal_token_df_tokens,timing_df,original_txt,story)
        adjusted_equal_token_df_tokens['event_len'] = adjusted_equal_token_df_tokens['event_ends']-adjusted_equal_token_df_tokens['event_starts']
        adjusted_equal_token_df_tokens = adjusted_equal_token_df_tokens.drop(['H_event_conditioned','num_boundaries','has_boundary'], axis=1)
        adjusted_equal_token_df_tokens = check_is_boundary(adjusted_equal_token_df_tokens,segmentation_indices_in_tokens,story)
        pairwise_event_save_dir_adjusted = os.path.join(args.save_dir,model_save_dir_name,'pairwise_event',story,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
        if not os.path.exists(pairwise_event_save_dir_adjusted):
            os.makedirs(pairwise_event_save_dir_adjusted)
        adjusted_equal_token_df_tokens.to_csv(os.path.join(pairwise_event_save_dir_adjusted,'story_split_by_token_df_adjusted.csv'),index = False)
    else: # split by duration
        if args.parse_adjusted:
            if args.factor is not None:
                adjusted_equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration_factor_%.1f_adjusted.csv'%args.factor))
                adjusted_df_equal_duration_tokens = align_to_token_indices(adjusted_equal_duration_df_text,original_txt,tokenized_txt,story_tokens,tokenizer,args.model)
                pairwise_event_save_dir_adjusted = os.path.join(args.save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                adjusted_equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration_adjusted.csv'))
                adjusted_df_equal_duration_tokens = align_to_token_indices(adjusted_equal_duration_df_text,original_txt,tokenized_txt,story_tokens,tokenizer,args.model)
                pairwise_event_save_dir_adjusted = os.path.join(args.save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing_adjusted')
            if not os.path.exists(pairwise_event_save_dir_adjusted):
                os.makedirs(pairwise_event_save_dir_adjusted)
            adjusted_df_equal_duration_tokens.to_csv(os.path.join(pairwise_event_save_dir_adjusted,'story_split_by_duration_df_adjusted.csv'),index = False)
        else:
            # the original recombined events
            if args.factor is not None:
                equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration_factor_%.1f.csv'%args.factor))
                df_equal_duration_tokens = align_to_token_indices(equal_duration_df_text,original_txt,tokenized_txt,story_tokens,tokenizer,args.model)
                pairwise_event_save_dir = os.path.join(args.save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing_factor_%.1f'%args.factor)
            else:
                equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration.csv'))
                df_equal_duration_tokens = align_to_token_indices(equal_duration_df_text,original_txt,tokenized_txt,story_tokens,tokenizer,args.model)
                pairwise_event_save_dir = os.path.join(args.save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing')
            if not os.path.exists(pairwise_event_save_dir):
                os.makedirs(pairwise_event_save_dir)
            df_equal_duration_tokens.to_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'),index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--story")
    parser.add_argument("--save_dir",default='../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--timing_dir",default = "../behavior_data/transcripts/timing",help = "directory storing timing dfs")
    parser.add_argument("--event_timing_dir", default = '../behavior_data/story_split_timing')
    parser.add_argument("--parse_adjusted",action ='store_true',help = 'redo the timing for the manually adjusted split')
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks')
    parser.add_argument("--even_tokens",action='store_true', help = 'splitted by even tokens, not even duration')

    args = parser.parse_args()
    main(args)