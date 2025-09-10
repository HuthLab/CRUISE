# calculate mutual information 
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
from utils import model_to_path_dict

def parse_recall_event_concat(save_dir,subjects):
    with open(os.path.join(save_dir,'event_only_stim.pkl'),'rb') as f:
        event_only_stim_list = pickle.load(f)
    with open(os.path.join(save_dir,'recall_event_stim.pkl'),'rb') as f:
        recall_event_stim_dict = pickle.load(f)
    with open(os.path.join(save_dir,'event_only_ce.pkl'),'rb') as f:
        event_only_ce = pickle.load(f)
    with open(os.path.join(save_dir,'recall_event_ce.pkl'),'rb') as f:
        recall_event_ce_dict = pickle.load(f)
    all_events = []
    all_subjects = []
    all_ce_diff = []
    for event_idx,this_event_ce in enumerate(event_only_ce):
        event_tokens = event_only_stim_list[event_idx]
        assert event_tokens.shape[1]-1 ==this_event_ce.shape[0],'tokens should be 1 longer than its CE, cuz BOS token at the start'

        this_recall_event_ce = recall_event_ce_dict[event_idx]
        this_recall_event_stim = recall_event_stim_dict[event_idx]
        this_concat_event_start_indices = this_recall_event_stim['event_start_indices']
        this_concat_event_end_indices = this_recall_event_stim['event_end_indices']
        this_concat_tokens = this_recall_event_stim['input_tokens']
        for subject_idx in range(len(subjects)):
            subject = subjects[subject_idx]
            subject_recall_event_ce = this_recall_event_ce[subject_idx]
            subject_concat_tokens = this_concat_tokens[subject_idx]
            assert subject_concat_tokens.shape[1]-1 ==subject_recall_event_ce.shape[0],'tokens should be 1 longer than its CE, cuz BOS token at the start'
            concat_event_start_idx = this_concat_event_start_indices[subject_idx]-1
            concat_event_end_idx = this_concat_event_end_indices[subject_idx]-1
            subject_event_ce = subject_recall_event_ce[concat_event_start_idx:concat_event_end_idx]
            if this_event_ce.shape != subject_event_ce.shape:
                print(this_event_ce.shape,subject_event_ce.shape)
            assert this_event_ce.shape == subject_event_ce.shape,'original and post-concat event CE shape must be the same'
            ce_diff = -torch.sum(subject_event_ce-this_event_ce).numpy()
            all_events.append(event_idx)
            all_subjects.append(subject)
            all_ce_diff.append(ce_diff)
    recall_explained_info_df = pd.DataFrame({'event':all_events,
                                             'subject':all_subjects,
                                             'ER_intersect':np.array(all_ce_diff)})
    recall_explained_info_df.to_csv(os.path.join(save_dir,'recall_explained_event_ce_df.csv'),index = False)


def parse_event_recall_concat(save_dir,num_events):
    with open(os.path.join(save_dir,'recall_only_stim.pkl'),'rb') as f:
        recall_only_stim_dict = pickle.load(f)
    with open(os.path.join(save_dir,'recall_only_ce.pkl'),'rb') as f:
        recall_only_ce = pickle.load(f)
    with open(os.path.join(save_dir,'event_recall_stim.pkl'),'rb') as f:
        event_recall_stim_dict = pickle.load(f)
    with open(os.path.join(save_dir,'event_recall_ce.pkl'),'rb') as f:
        event_recall_ce_dict = pickle.load(f)
    subjects = recall_only_stim_dict['subject_id']
    all_events = []
    all_subjects = []
    all_ce_diff = []
    for sub_idx,this_recall_ce in enumerate(recall_only_ce):
        subject = subjects[sub_idx]
        recall_tokens = recall_only_stim_dict['input_tokens'][sub_idx]
        assert recall_tokens.shape[1]-1 ==this_recall_ce.shape[0],'tokens should be 1 longer than its CE, cuz BOS token at the start'
        sub_recall_only_ce = this_recall_ce
        sub_event_recall_ce = event_recall_ce_dict[sub_idx]
        sub_event_recall_stim = event_recall_stim_dict[sub_idx]
        sub_concat_recall_start_indices = sub_event_recall_stim['recall_start_indices']
        sub_concat_recall_end_indices = sub_event_recall_stim['recall_end_indices']
        sub_concat_tokens = sub_event_recall_stim['input_tokens']
        for event_idx in range(num_events):
            this_event_recall_ce = sub_event_recall_ce[event_idx]
            this_concat_tokens = sub_concat_tokens[event_idx]
            assert this_concat_tokens.shape[1]-1 ==this_event_recall_ce.shape[0],'tokens should be 1 longer than its CE, cuz BOS token at the start'
            concat_recall_start_idx = sub_concat_recall_start_indices[event_idx]-1 # -1 because the tokens indices have bos 
            concat_recall_end_idx = sub_concat_recall_end_indices[event_idx]-1
            post_concat_recall_ce = this_event_recall_ce[concat_recall_start_idx:concat_recall_end_idx]
            assert sub_recall_only_ce.shape == post_concat_recall_ce.shape,'original and post-concat recall CE shape must be the same'
            ce_diff = -torch.sum(post_concat_recall_ce-sub_recall_only_ce).numpy()
            all_events.append(event_idx)
            all_subjects.append(subject)
            all_ce_diff.append(ce_diff)
    recall_explained_info_df = pd.DataFrame({'event':all_events,
                                             'subject':all_subjects,
                                             'ER_intersect':np.array(all_ce_diff)})
    recall_explained_info_df.to_csv(os.path.join(save_dir,'event_explained_recall_ce_df.csv'),index = False)



def main(args):
    story = args.story
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])

    pairwise_event_save_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'pairwise_event',story)

    if args.split_story_by_duration:
        if args.adjusted:
            if args.factor is not None:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f_adjusted'%args.factor)
            else:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_adjusted')
            even_duration_split_df = pd.read_csv(os.path.join(even_split_save_dir,'story_split_by_duration_df_adjusted.csv'))
        else:
            if args.factor is not None:
                even_split_save_dir = pairwise_event_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
            even_duration_split_df = pd.read_csv(os.path.join(even_split_save_dir,'story_split_by_duration_df.csv'))

        save_dir = even_split_save_dir
        num_events = len(even_duration_split_df)
    elif args.split_story_by_tokens:
        if args.adjusted:
            if args.factor is not None:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f_adjusted'%args.factor)
            else:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_adjusted')
            even_token_split_df = pd.read_csv(os.path.join(even_split_save_dir,'story_split_by_token_df_adjusted.csv'))
        else:
            if args.factor is not None:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens_factor_%.1f'%args.factor)
            else:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_tokens')
            even_token_split_df = pd.read_csv(os.path.join(even_split_save_dir,'story_split_by_token_df.csv'))
        save_dir = even_split_save_dir
        num_events = len(even_token_split_df)

    else:
        save_dir = pairwise_event_save_dir
        if args.story =='sherlock':
            if args.twosessions:
                sherlock_event_df = pd.read_csv(os.path.join(args.sherlock_transcript_dir,'chen_scene_timing_sec_lines_noopening.csv'))
            else:
                sherlock_event_df = pd.read_csv(os.path.join(args.sherlock_transcript_dir,'truncated_scene_timing_sec.csv'))
            num_events = len(sherlock_event_df)
        else:
            # load consensus
            consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%args.story)
            with open(consensus_path,'r') as f:
                consensus_txt = f.read()
            consensus_txt = consensus_txt.split('\n')
            num_events = len(consensus_txt)
    
    if args.random_recalls:
        save_dir = os.path.join(save_dir,'random_recalls')
    # load recall transcripts 
    recall_transcript_dir = os.path.join(args.recall_transcript_dir,story)

    if args.story =='sherlock' and args.twosessions:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_corrected_recall_transcripts_2sessions.csv'%story))
    else:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_corrected_recall_transcripts.csv'%story))
    corrected_transcript = corrected_transcript.dropna(axis = 0) # drop bad subjects (nan in corrected transcript)
    subjects = corrected_transcript['subject'].astype(int)

    if args.recall_event_concat:
        parse_recall_event_concat(save_dir,subjects)
    if args.event_recall_concat:
        parse_event_recall_concat(save_dir,num_events)
    

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