'''generate the uniform_encoding_df_even_split_adjusted.csv for even split events. 
right now doesn't work for the regular uniform encoding! only the even split
'''
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import numpy as np
import pickle
import glob
import re
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances
import string
import statsmodels.api as sm
import itertools
import sys
from utils import get_segmentation_indices,segmentation_to_word_list
from utils import calculate_cross_entropy,normalize_entropy,model_to_path_dict
import argparse

def get_pairwise_event_explained_ce(pairwise_event_save_dir):
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_tokens.pkl'),'rb') as f:
        token_pairs_dict = pickle.load(f)
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_startidx_of_second_event.pkl'),'rb') as f:
        startidx_of_second_event_dict = pickle.load(f)
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_ce.pkl'),'rb') as f:
        pairwise_event_ce_dict = pickle.load(f)
    with open(os.path.join(pairwise_event_save_dir,'event_only_ce.pkl'),'rb') as f:
        all_events_only_ce = pickle.load(f)
    pairwise_event_explained_ce_dict = {} # I(first event;second event)
    for event_pair in pairwise_event_ce_dict.keys():
        second_event = event_pair[1]
        pairwise_event_ce = pairwise_event_ce_dict[event_pair]
        startidx_of_second_event = startidx_of_second_event_dict[event_pair]
        second_event_ce = pairwise_event_ce[startidx_of_second_event:]
        assert second_event_ce.shape == all_events_only_ce[second_event].shape, 'CE shape of the same event should be the same'
        explained_ce = -torch.sum(second_event_ce-all_events_only_ce[second_event]).numpy()
        pairwise_event_explained_ce_dict[event_pair] = explained_ce
    return pairwise_event_explained_ce_dict

def get_conditional_uniform_encoding(event_len,all_events_only_ce,conditional_event_ce,pairwise_event_explained_ce_dict):
    all_events_sum_conditional = [] # the sigma term
    for event_i in range(len(event_len)):
        event_i_sum = 0
        event_i_info = torch.sum(all_events_only_ce[event_i])
        event_i_conditional_ce = conditional_event_ce[event_i]
        for event_j in range(len(event_len)):
            event_j_info = torch.sum(all_events_only_ce[event_j])
            event_j_conditional_ce = conditional_event_ce[event_j]
            if event_i == event_j:
                interaction = event_i_info
            else:
                event_pair = [event_i,event_j]
                event_pair.sort()
                interaction = pairwise_event_explained_ce_dict[tuple(event_pair)]
            product = interaction/event_j_info*event_j_conditional_ce
            event_i_sum+=product
        all_events_sum_conditional.append(event_i_sum.numpy())
    all_events_sum_conditional = np.array(all_events_sum_conditional)
    return all_events_sum_conditional

def get_event_len_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict):
    all_events_sum = []
    for event_i in range(len(event_len)):
        event_i_sum = 0
        event_i_info = torch.sum(all_events_only_ce[event_i])
        event_i_len = event_len[event_i]
        for event_j in range(len(event_len)):
            event_j_info = torch.sum(all_events_only_ce[event_j])
            event_j_len = event_len[event_j]
            if event_i == event_j:
                interaction = event_i_info
            else:
                event_pair = [event_i,event_j]
                event_pair.sort()
                interaction = pairwise_event_explained_ce_dict[tuple(event_pair)]
            product = interaction/event_j_info*event_j_len
            event_i_sum+=product
        all_events_sum.append(event_i_sum.numpy())
    all_events_sum = np.array(all_events_sum)
    return all_events_sum

def get_duration_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict,event_duration):
    all_events_sum_by_duration = []
    for event_i in range(len(event_len)):
        event_i_sum = 0
        event_i_info = torch.sum(all_events_only_ce[event_i])
        for event_j in range(len(event_len)):
            event_j_info = torch.sum(all_events_only_ce[event_j])
            event_j_duration = event_duration[event_j]
            if event_i == event_j:
                interaction = event_i_info
            else:
                event_pair = [event_i,event_j]
                event_pair.sort()
                interaction = pairwise_event_explained_ce_dict[tuple(event_pair)]
            product = interaction/event_j_info*event_j_duration
            event_i_sum+=product
        all_events_sum_by_duration.append(event_i_sum.numpy())
    all_events_sum_by_duration = np.array(all_events_sum_by_duration)
    return all_events_sum_by_duration

def main(args):
    story = args.story
    adjusted = args.adjusted
    save_dir = args.save_dir
    model_name = args.model
    model_save_dir_name = model_to_path_dict[model_name]['save_dir_name']
    model_initial_char = model_to_path_dict[model_name]['initial_char']
    if model_name =='Llama3.2-3b-instruct_finetuned':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[model_name]['hf_name'])
    print(save_dir)
    print('ls save_dir:',os.listdir(save_dir))
    moth_output_dir = os.path.join(save_dir,model_save_dir_name,'moth_stories_output')
    with open(os.path.join(moth_output_dir,story,'cross_entropy.pkl'),'rb') as f:
        original_ce = pickle.load(f)
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
    
    exclusion_df = pd.read_csv(os.path.join(args.exclusion_dir,'%s_exclusion.csv'%story))
    excluded_subjects = exclusion_df['excluded'].loc[exclusion_df['recall']==1].values
    timing_df = pd.read_csv(os.path.join(args.timing_dir,'%s_timing.csv'%story))

    if adjusted:
        pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing_adjusted')
        story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df_adjusted.csv'))
    else:
        pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,'story_split_timing')
        story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'))
    event_len=story_split_by_duration_df['event_ends'] - story_split_by_duration_df['event_starts']

    pairwise_event_explained_ce_dict = get_pairwise_event_explained_ce(pairwise_event_save_dir)

    # direct concatenation
    if args.random_recalls:
        recall_explained_event_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'random_recalls','recall_explained_event_ce_df.csv'))
    else:
        recall_explained_event_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'recall_explained_event_ce_df.csv'))
        recall_explained_event_info_df = recall_explained_event_info_df[~recall_explained_event_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect = recall_explained_event_info_df.groupby('event')['ER_intersect'].mean()
    assert len(consensus_txt)==len(mean_ER_intersect)

    if args.random_recalls:
        event_explained_recall_info_df_raw = pd.read_csv(os.path.join(pairwise_event_save_dir,'random_recalls','event_explained_recall_ce_df.csv'))
    else:
        event_explained_recall_info_df_raw = pd.read_csv(os.path.join(pairwise_event_save_dir,'event_explained_recall_ce_df.csv'))
        event_explained_recall_info_df_raw = event_explained_recall_info_df_raw[~event_explained_recall_info_df_raw['subject'].isin(excluded_subjects)]
    mean_ER_intersect_recall_last = event_explained_recall_info_df_raw.groupby('event')['ER_intersect'].mean()
    assert len(consensus_txt)==len(mean_ER_intersect_recall_last)

    # instruct
    if args.random_recalls:
        recall_explained_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'random_recalls','instruct','recall_explained_event_ce_df.csv'))
    else:
        recall_explained_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'instruct','recall_explained_event_ce_df.csv'))
        recall_explained_info_df = recall_explained_info_df[~recall_explained_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect_instruct = recall_explained_info_df.groupby('event')['ER_intersect'].mean()
    assert len(consensus_txt)==len(mean_ER_intersect_instruct)

    if args.random_recalls:
        event_explained_recall_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'random_recalls','instruct','event_explained_recall_ce_df.csv'))
    else:
        event_explained_recall_info_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'instruct','event_explained_recall_ce_df.csv'))
        event_explained_recall_info_df = event_explained_recall_info_df[~event_explained_recall_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect_recall_last_instruct = event_explained_recall_info_df.groupby('event')['ER_intersect'].mean()
    assert len(consensus_txt)==len(mean_ER_intersect_recall_last_instruct)

    # conditional CE
    conditional_event_ce = []
    for idx in range(len(story_split_by_duration_df)):
        start_idx = story_split_by_duration_df['event_starts'].iloc[idx]
        end_idx = story_split_by_duration_df['event_ends'].iloc[idx]
        conditional_event_ce.append(torch.sum(original_ce[start_idx:end_idx]))
    conditional_event_ce = np.array(conditional_event_ce)

    with open(os.path.join(pairwise_event_save_dir,'event_only_ce.pkl'),'rb') as f:
        all_events_only_ce = pickle.load(f)
    # uniform encoding, conditional
    all_events_sum_conditional = get_conditional_uniform_encoding(event_len,all_events_only_ce,conditional_event_ce,pairwise_event_explained_ce_dict)
    all_events_sum = get_event_len_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict)

    event_duration = story_split_by_duration_df['Duration']
    all_events_sum_by_duration = get_duration_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict,event_duration)

    all_events_only_sum_ce = np.array([torch.sum(t) for t in all_events_only_ce])

    uniform_encoding_save_dir = os.path.join(save_dir,model_save_dir_name,'uniform_encoding',story)
    if not os.path.exists(uniform_encoding_save_dir):
        os.makedirs(uniform_encoding_save_dir)
    uniform_encoding_df = pd.DataFrame({'H(event|prev events)':conditional_event_ce,
                                        'H(event)':all_events_only_sum_ce,
                                        'event_len':event_len,
                                        'event_duration':event_duration,
                                        'weighted_event_info':all_events_sum,
                                        'weighted_event_info_by_duration':all_events_sum_by_duration,
                                        'weighted_event_info_conditioned':all_events_sum_conditional,
    #                                     'mean_correct_details':mean_correct_details,
    #                                     'mean_cosine_similarity':mean_cosine_sim_by_event,
                                        'mean_ER_intersect':mean_ER_intersect,
                                        'mean_ER_intersect_recall_last':mean_ER_intersect_recall_last,
                                        'mean_ER_intersect_instruct':mean_ER_intersect_instruct,
                                        'mean_ER_intersect_instruct_recall_last':mean_ER_intersect_recall_last_instruct
                            })
    if adjusted:
        if args.random_recalls:
            uniform_encoding_df.to_csv(os.path.join(uniform_encoding_save_dir,'uniform_encoding_df_even_split_adjusted_random_recalls.csv'),index = False)
        else:
            uniform_encoding_df.to_csv(os.path.join(uniform_encoding_save_dir,'uniform_encoding_df_even_split_adjusted.csv'),index = False)
    else:
        uniform_encoding_df.to_csv(os.path.join(uniform_encoding_save_dir,'uniform_encoding_df_even_split.csv'),index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir",default='../data/generated')
    parser.add_argument("--segmentation_dir",default = '../data/behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../data/behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--exclusion_dir",default = "../data/behavior_data/exclusion",help = "directory storing exclusion dfs")
    parser.add_argument("--timing_dir",default = "../data/behavior_data/transcripts/timing",help = "directory storing timing dfs")
    parser.add_argument("--story",default = 'pieman')
    parser.add_argument("--model")
    parser.add_argument("--adjusted",action = 'store_true',help = 'whether to use adjusted even split')
    parser.add_argument("--random_recalls",action = 'store_true',help = 'whether to use randomly sampled recalls from other stories')
    args = parser.parse_args()
    main(args)