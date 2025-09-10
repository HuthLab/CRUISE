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

def get_split_chunk_txt(equal_duration_df,timing_df,original_txt,story):
    '''Find the text that corresponds to the timing of each chunk'''
    # onset:offset gets you the words in the recombined event
    all_onset_idx_in_story = [] # word index in timing_df
    for i,row in equal_duration_df.iterrows():
        onset = row['Onset']
        if story =='pieman':
            if onset > timing_df['start'].loc[len(timing_df)-1]/1000:
                onset_idx_in_story = len(timing_df)-1
            else:
                onset_idx_in_story = np.where(timing_df['start']/1000-onset>=0)[0][0]
        else:
            onset_idx_in_story = np.where(timing_df['start']-onset>=0)[0][0]

        all_onset_idx_in_story.append(onset_idx_in_story)
    all_onset_idx_in_story = np.array(all_onset_idx_in_story)
    all_offset_idx_in_story = all_onset_idx_in_story[1:]
    all_offset_idx_in_story = np.concatenate([all_offset_idx_in_story,[len(timing_df)]])

    all_split_txt = []
    for i in range(len(equal_duration_df)):
        start_idx = all_onset_idx_in_story[i]
        end_idx = all_offset_idx_in_story[i]
        chunk_txt = timing_df['text'].iloc[start_idx:end_idx]
        chunk_txt = [t if isinstance(t,str) else '' for t in list(chunk_txt.values)]
        recombined_txt = ' '.join(chunk_txt)
        recombined_txt = recombined_txt.lower()
        assert recombined_txt in original_txt
        all_split_txt.append(recombined_txt)
    equal_duration_df['text'] = all_split_txt
    return equal_duration_df

def split_story_by_duration(timing_df,num_events,story):
    story_start = timing_df['start'].iloc[0]
    story_end = timing_df['stop'].iloc[len(timing_df)-1]
    chunk_duration = (story_end-story_start)/num_events
    
    chunk_starts = []
    chunk_ends = []
    for i in range(num_events):
        if i ==0:
            chunk_start = story_start
        if i == num_events:
            chunk_end = story_end
        else:
            chunk_end = chunk_start+chunk_duration
        chunk_starts.append(chunk_start)
        chunk_ends.append(chunk_end)
        chunk_start = chunk_end
    if story=='pieman':
        chunk_starts = np.array(chunk_starts)/1000
        chunk_ends = np.array(chunk_ends)/1000
        chunk_duration/=1000
    df = pd.DataFrame({'Onset':chunk_starts,'Offset':chunk_ends,'Duration':chunk_duration})
    return df

def main(args):
    story = args.story
    model_initial_char = model_to_path_dict[args.model]['initial_char']
    model_save_dir_name = model_to_path_dict[args.model]['save_dir_name']
    moth_output_dir = os.path.join(args.save_dir,model_save_dir_name,'moth_stories_output')
    # tokenized txt 
    with open(os.path.join(moth_output_dir,story,'tokenized_txt.pkl'),'rb') as f:
        tokenized_txt = pickle.load(f)

    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
        original_txt = f.read()

    timing_df = pd.read_csv(os.path.join(args.timing_dir,'%s_timing.csv'%story))

    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char=model_initial_char)
    event_len = np.diff([0]+segmentation_indices_in_tokens)
    event_len[0]+=1

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
    num_events = len(consensus_txt)

    event_timing_dir = os.path.join(args.event_timing_dir,story)
    if not os.path.exists(event_timing_dir):
        os.makedirs(event_timing_dir)
    if args.parse_adjusted:
        if args.factor is not None:
            adjusted_equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration_factor_%.1f_adjusted.csv'%args.factor))
            adjusted_equal_duration_df_text = align_timing_with_text(adjusted_equal_duration_df_text,timing_df,original_txt,story)
            adjusted_equal_duration_df_text.to_csv(os.path.join(event_timing_dir,'story_even_duration_factor_%.1f_adjusted.csv'%args.factor),index=False)
        else:
            adjusted_equal_duration_df_text = pd.read_csv(os.path.join(event_timing_dir,'story_even_duration_adjusted.csv'))
            adjusted_equal_duration_df_text = align_timing_with_text(adjusted_equal_duration_df_text,timing_df,original_txt,story)
            adjusted_equal_duration_df_text.to_csv(os.path.join(event_timing_dir,'story_even_duration_adjusted.csv'),index=False)
    else:
        if args.factor is not None:
            equal_duration_df = split_story_by_duration(timing_df,int(num_events*args.factor),story)
        else:
            equal_duration_df = split_story_by_duration(timing_df,num_events,story)
        equal_duration_df_text = get_split_chunk_txt(equal_duration_df,timing_df,original_txt,story)
        equal_duration_df_text = align_timing_with_text(equal_duration_df_text,timing_df,original_txt,story)
        if args.factor is not None:
            equal_duration_df_text.to_csv(os.path.join(event_timing_dir,'story_even_duration_factor_%.1f.csv'%args.factor),index=False)
        else:
            equal_duration_df_text.to_csv(os.path.join(event_timing_dir,'story_even_duration.csv'),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir",default = '../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--timing_dir",default = "../behavior_data/transcripts/timing",help = "directory storing timing dfs")
    parser.add_argument("--event_timing_dir", default = '../behavior_data/story_split_timing')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--story")
    parser.add_argument("--parse_adjusted",action ='store_true',help = 'redo the timing for the manually adjusted split')
    parser.add_argument("--factor",type=float, help = 'multiplication factor, creates int(factor * #original events) total chunks')
    args = parser.parse_args()
    main(args)