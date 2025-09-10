# tokenize transcripts for later inference
from transformers import AutoTokenizer
import torch 
import numpy as np
import pandas as pd
import pickle
import accelerate
import os
import re 
import glob
from tqdm import tqdm
from torch.nn import functional as F
import argparse
import string
from utils import get_segmentation_indices,segmentation_to_word_list,model_to_path_dict

def get_recall_tokens(tokenizer,corrected_transcript):
    '''output dictionary with 2 entries: subject_id and input_tokenized,
    contains bos tokens, has the batch dimension'''
    recall_tokens_dict = {}
    subject_ids = []
    recall_tokens = []
    for i in tqdm(corrected_transcript.index):
        subject_id = corrected_transcript['subject'][i]
        transcript = corrected_transcript['corrected transcript'][i]
        remove_punctuation = string.punctuation.translate(str.maketrans('', '', '\'')) # remove all punctuation except ' cuz abbreviations
        no_punctuation_transcript = transcript.translate(str.maketrans('', '', remove_punctuation))
        no_punctuation_transcript = no_punctuation_transcript.lower()
        if no_punctuation_transcript.startswith(' '):
            no_punctuation_transcript = no_punctuation_transcript[1:]
        if no_punctuation_transcript.endswith(' '):
            no_punctuation_transcript = no_punctuation_transcript[:-1]
        
        recall_tokenized = tokenizer(no_punctuation_transcript, return_tensors="pt").input_ids # 1 * n tokens
        subject_ids.append(subject_id)
        recall_tokens.append(recall_tokenized)
    recall_tokens_dict['subject_id'] = subject_ids
    recall_tokens_dict['input_tokens'] = recall_tokens
    return recall_tokens_dict

def get_event_tokens(tokenizer,story_tokens,event_start_indices,event_end_indices):
    '''output lists of tensors, contains bos tokens, has the batch dimension'''
    bos_token_tensor = torch.Tensor([tokenizer.bos_token_id]).type(torch.int64) # int64 is longTensor
    all_event_tokens = []
    for i in range(len(event_start_indices)):
        event_start_idx = event_start_indices[i]
        event_end_idx = event_end_indices[i]
        event_tokens = story_tokens[event_start_idx:event_end_idx]
        event_tokens = torch.concat((bos_token_tensor,event_tokens))
        event_tokens = torch.unsqueeze(event_tokens,0) # 1 * n tokens
        all_event_tokens.append(event_tokens)
    return all_event_tokens

def get_event_recall_tokens(recall_tokens_dict,all_event_tokens):
    '''outputs dict of dicts. first key is subject index'''
    concat_stim_dict = {}
    for i in tqdm(range(len(recall_tokens_dict['subject_id']))):
        subject_concat_tokens = []
        # [start:end] gives that chunk
        event_start_indices = []
        event_end_indices = []
        recall_start_indices = []
        recall_end_indices = []
        subject_id = recall_tokens_dict['subject_id'][i]
        recall_tokens = recall_tokens_dict['input_tokens'][i]
        recall_tokens = recall_tokens[:,1:] # 1 * n tokens
        for event_idx in range(len(all_event_tokens)):
            event_tokens = all_event_tokens[event_idx]
            concat_tokens = torch.cat([event_tokens,recall_tokens],dim=1)
            event_start_indices.append(1) # cuz of bos token
            event_end_indices.append(event_tokens.shape[1])
            recall_start_indices.append(event_tokens.shape[1])
            recall_end_indices.append(concat_tokens.shape[1])
            subject_concat_tokens.append(concat_tokens)

        concat_stim_dict[i] = {'input_tokens':subject_concat_tokens,
                                'event_start_indices':event_start_indices,
                                'event_end_indices':event_end_indices,
                                'recall_start_indices':recall_start_indices,
                                'recall_end_indices':recall_end_indices,
                                'subject':subject_id
                                }
    return concat_stim_dict

def get_recall_event_tokens(recall_tokens_dict,all_event_tokens):
    '''outputs dict of dicts. first key is event index'''
    concat_stim_dict = {}
    for event_idx in range(len(all_event_tokens)):
        event_tokens = all_event_tokens[event_idx]
        event_tokens = event_tokens[:,1:] # 1 * n tokens
        event_concat_tokens = []
        # [start:end] gives that chunk
        event_start_indices = []
        event_end_indices = []
        recall_start_indices = []
        recall_end_indices = []
        subjects = []
        for sub_idx in tqdm(range(len(recall_tokens_dict['subject_id']))):
            subject = recall_tokens_dict['subject_id'][sub_idx]
            recall_tokens = recall_tokens_dict['input_tokens'][sub_idx]
            concat_tokens = torch.cat([recall_tokens,event_tokens],dim=1)
            recall_start_indices.append(1)
            recall_end_indices.append(recall_tokens.shape[1])
            event_start_indices.append(recall_tokens.shape[1])
            event_end_indices.append(concat_tokens.shape[1])
            event_concat_tokens.append(concat_tokens)
            subjects.append(subject)
    
        concat_stim_dict[event_idx] = {'input_tokens':event_concat_tokens,
                                        'event_start_indices':event_start_indices,
                                        'event_end_indices':event_end_indices,
                                        'recall_start_indices':recall_start_indices,
                                        'recall_end_indices':recall_end_indices,
                                        'subject':subjects
                                    }
    return concat_stim_dict
def main(args):
    story = args.story
    model_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'])
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    

    moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'moth_stories_output')
    # load consensus
    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%args.story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    consensus_wordlist = segmentation_to_word_list(consensus_txt)
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
    if not os.path.exists(pairwise_event_save_dir):
        os.makedirs(pairwise_event_save_dir)

    if args.split_story_by_duration:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_adjusted')
            even_duration_split_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df_adjusted.csv'))
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
            even_duration_split_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'))
        event_txt =  even_duration_split_df['text'].values
        event_start_indices = even_duration_split_df['event_starts']
        event_end_indices = even_duration_split_df['event_ends']
    elif args.split_story_by_tokens:
        if args.adjusted:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_adjusted')
            even_token_split_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df_adjusted.csv'))
        else:
            if args.factor is not None:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f'%args.factor)
            else:
                pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens')
            even_token_split_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df.csv'))
        event_txt =  even_token_split_df['text'].values
        event_start_indices = even_token_split_df['event_starts']
        event_end_indices = even_token_split_df['event_ends']

    # load recall transcripts 
    recall_transcript_dir = os.path.join(args.recall_transcript_dir,story)
    if args.random_recalls:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_random_recall_transcripts.csv'%story))
    else:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_corrected_recall_transcripts.csv'%story))
    subjects = corrected_transcript['subject'].values
    corrected_transcript = corrected_transcript.dropna(axis = 0) # drop bad subjects (nan in corrected transcript)
    print('num recalls',len(corrected_transcript))
    
    if args.random_recalls:
        save_dir = os.path.join(pairwise_event_save_dir,'random_recalls')
    else:
        save_dir = pairwise_event_save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    recall_tokens_dict = get_recall_tokens(tokenizer,corrected_transcript)
    all_event_tokens = get_event_tokens(tokenizer,story_tokens,event_start_indices,event_end_indices)
    if args.recall_event_concat:
        recall_event_stim_dict = get_recall_event_tokens(recall_tokens_dict,all_event_tokens)
        with open(os.path.join(save_dir,'recall_event_stim.pkl'),'wb') as f:
            pickle.dump(recall_event_stim_dict,f)
    if args.event_recall_concat:
        event_recall_stim_dict = get_event_recall_tokens(recall_tokens_dict,all_event_tokens)
        with open(os.path.join(save_dir,'event_recall_stim.pkl'),'wb') as f:
            pickle.dump(event_recall_stim_dict,f)
    with open(os.path.join(save_dir,'event_only_stim.pkl'),'wb') as f:
        pickle.dump(all_event_tokens,f)
    with open(os.path.join(save_dir,'recall_only_stim.pkl'),'wb') as f:
        pickle.dump(recall_tokens_dict,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir", default = "/home/jianing/generation/generated/")
    parser.add_argument("--segmentation_dir",default = '/home/jianing/generation/behavior_data/segmentation')
    parser.add_argument("--recall_transcript_dir",default = '/home/jianing/generation/behavior_data/recall_transcript')
    parser.add_argument("--original_transcript_dir",default='/home/jianing/generation/transcripts/moth_stories')
    parser.add_argument("--moth_output_dir",default = '/home/jianing/generation/generated/')
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