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
from utils import calculate_cross_entropy,normalize_entropy,model_to_path_dict
import argparse

def get_event_timing_df(story,idx_of_boundaries,consensus_wordlist,timing_df,num_events):
    all_events_start_time = []
    all_events_end_time = []
    for count,idx in enumerate(idx_of_boundaries):
        row = timing_df.iloc[idx]
        assert consensus_wordlist[idx].strip().lower() == row['text'].strip().lower()
        if count == 0:
            event_start_time = timing_df['start'].iloc[0]
        else:
            event_start_time = timing_df['start'].iloc[idx_of_boundaries[count-1]+1] # next row after event ends
        event_end_time = row['stop']
        if story =='pieman':
            event_start_time/=1000
            event_end_time/=1000
        all_events_start_time.append(event_start_time)
        all_events_end_time.append(event_end_time)
    event_timing_df = pd.DataFrame({'Event':np.arange(num_events),
                                'Onset':all_events_start_time,
                                'Offset':all_events_end_time
                                })
    return event_timing_df

def split_and_recombine_event(event_timing_df,story,timing_df,original_txt):
    '''
    event_timing_df: Onset and offset of each event
    timing_df: Onset and offset of each word
    '''
    split_rows = []
    # Step 1: Split each scene into 4 parts
    for _, row in event_timing_df.iterrows():
        onset = row['Onset']
        offset = row['Offset']
        quarter1 = onset + (offset - onset) / 4
        quarter2 = onset + (offset - onset) / 2
        quarter3 = onset + 3 * (offset - onset) / 4

        # Add each quarter as a new row
        split_rows.append({
            'Scene': f"{row['Event']}_1",
            'Onset': onset,
            'Offset': quarter1
        })

        split_rows.append({
            'Scene': f"{row['Event']}_2",
            'Onset': quarter1,
            'Offset': quarter2
        })

        split_rows.append({
            'Scene': f"{row['Event']}_3",
            'Onset': quarter2,
            'Offset': quarter3
        })

        split_rows.append({
            'Scene': f"{row['Event']}_4",
            'Onset': quarter3,
            'Offset': offset
        })

    df_split = pd.DataFrame(split_rows)
    # Step 2: Recombine the segments 
    recombined_rows = []
    recombine_start_row = 0
    # Leave the first quarter of the first scene on its own
    recombined_rows.append({
        'recombined_scene': 0,
        'Onset': split_rows[recombine_start_row]['Onset'],
        'Offset': split_rows[recombine_start_row]['Offset']
    })

    # Recombine quarter 2 and 3 of the same scene, then quarter 4 and next scene's quarter 1
    for i in range(recombine_start_row, len(split_rows) - 4, 4):
        # Combine Quarter 2 and 3 of the current scene
        recombined_rows.append({
            'recombined_scene': len(recombined_rows),
            'Onset': split_rows[i+1]['Onset'],
            'Offset': split_rows[i+2]['Offset']
        })

        # Combine Quarter 4 of the current scene with Quarter 1 of the next scene
        recombined_rows.append({
            'recombined_scene': len(recombined_rows),
            'Onset': split_rows[i+3]['Onset'],
            'Offset': split_rows[i+4]['Offset']
        })
    if i + 4 < len(split_rows): # there should be two rows left in this scenario
        recombined_rows.append({
            'recombined_scene': len(recombined_rows),
            'Onset': split_rows[i+4+1]['Onset'],
            'Offset': split_rows[i+4+2]['Offset']
        })
    # Leave the last quarter of the last scene on its own
    recombined_rows.append({
        'recombined_scene': len(recombined_rows),
        'Onset': split_rows[-1]['Onset'],
        'Offset': split_rows[-1]['Offset']
    })

    # Create a new DataFrame with the recombined rows
    df_recombined = pd.DataFrame(recombined_rows)
    assert len(df_recombined)== 2*len(event_timing_df)+1

    # get the corresponding text of each recombined event
    # onset:offset gets you the words in the recombined event
    all_onset_idx_in_story = [] # word index in timing_df
    for i,row in df_recombined.iterrows():
        onset = row['Onset']
        if story =='pieman':
            if onset > timing_df['start'].loc[len(timing_df)-1]/1000:
                onset_idx_in_story = len(timing_df)-1
            else:
                onset_idx_in_story = np.where(timing_df['start']/1000-onset>=0)[0][0]
        else:
            # if story=='souls' and i >= len(df_recombined)-2:
            #     onset_idx_in_story = len(timing_df)-1
            # else:
            onset_idx_in_story = np.where(timing_df['start']-onset>=0)[0][0]

        all_onset_idx_in_story.append(onset_idx_in_story)
    all_onset_idx_in_story = np.array(all_onset_idx_in_story)
    all_offset_idx_in_story = all_onset_idx_in_story[1:]
    all_offset_idx_in_story = np.concatenate([all_offset_idx_in_story,[len(timing_df)]])

    all_recombined_txt = []
    for i in range(len(df_recombined)):
        start_idx = all_onset_idx_in_story[i]
        end_idx = all_offset_idx_in_story[i]
        chunk_txt = timing_df['text'].iloc[start_idx:end_idx]
        chunk_txt = [t if isinstance(t,str) else '' for t in list(chunk_txt.values)]
        recombined_txt = ' '.join(chunk_txt)
        recombined_txt = recombined_txt.lower()
        assert recombined_txt in original_txt
        all_recombined_txt.append(recombined_txt)
    df_recombined['text'] = all_recombined_txt
    df_recombined['Duration'] = df_recombined['Offset']-df_recombined['Onset']
    # if story =='souls': # last event of souls only has one word "you"
    #     df_recombined = df_recombined[:len(df_recombined)-2]
    df_recombined_aligned = align_timing_with_text(df_recombined,timing_df,original_txt,story)
    return df_recombined_aligned

def align_timing_with_text(df_recombined,timing_df,original_txt,story):
    '''
    realign the timing to text, because the actual onset and offset of the recombine events 
    can be different from the theoretical quarter division
    '''
    timing_word_list = [t if isinstance(t,str) else '' for t in list(timing_df['text'].values)]
    assert ' '.join(timing_word_list).lower() == original_txt
    chunk_duration = []
    chunk_start_time = []
    chunk_end_time = []
    word_list_start_idx = 0
    word_list_end_idx = 1
    for i,row in df_recombined.iterrows():
        chunk_txt = row['text']
        assert original_txt.find(chunk_txt)!=-1
        joined_text = ' '.join(timing_word_list[word_list_start_idx:word_list_end_idx]).lower()
        if len(joined_text)>0 and joined_text[0] == ' ':
            joined_text = joined_text[1:]
        if chunk_txt[0] == ' ':
            chunk_txt = chunk_txt[1:]
        if chunk_txt[-1] == ' ':
            chunk_txt = chunk_txt[:-1]
        if len(joined_text)>0:
            while joined_text[0] != chunk_txt[0]: #while joined_text.split()[0] != chunk_txt.split()[0]:
                word_list_start_idx+=1
                word_list_end_idx = word_list_start_idx+1
                joined_text = ' '.join(timing_word_list[word_list_start_idx:word_list_end_idx]).lower()
                if len(joined_text)>0 and joined_text[0] == ' ':
                    joined_text = joined_text[1:]
        while joined_text!= chunk_txt:
            word_list_end_idx+=1
            if word_list_end_idx > len(timing_df):
                print('not found for interval number %d'%i)
                break
            joined_text = ' '.join(timing_word_list[word_list_start_idx:word_list_end_idx]).lower()
            if joined_text[0] == ' ':
                joined_text = joined_text[1:]

        timing_start = timing_df['start'].iloc[word_list_start_idx]
        timing_end = timing_df['stop'].iloc[word_list_end_idx-1]
        if story =='pieman':
            timing_start/=1000
            timing_end/=1000
        chunk_start_time.append(timing_start)
        chunk_end_time.append(timing_end)
        chunk_duration.append(timing_end-timing_start)
        # update word list indices
        word_list_start_idx = word_list_end_idx
        word_list_end_idx = word_list_start_idx+1
    df_recombined_aligned = df_recombined.copy()
    df_recombined_aligned['Onset'] = chunk_start_time
    df_recombined_aligned['Offset'] = chunk_end_time
    df_recombined_aligned['Duration'] = chunk_duration
    return df_recombined_aligned

def main(args):
    story = args.story
    model_save_dir_name = model_to_path_dict[args.model]['save_dir_name']
    model_initial_char = model_to_path_dict[args.model]['initial_char']
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])

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

    if story =='pieman':
        timing_df = pd.read_csv('/home/jianing/generation/transcripts/pieman_timing.csv')
    else:
        timing_df = pd.read_csv(os.path.join(args.timing_dir,'%s_timing.csv'%story))

    consensus_wordlist = segmentation_to_word_list(consensus_txt)
    segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char='Ġ')
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
    idx_of_boundaries = [i for i,w in enumerate(consensus_wordlist) if '\n' in w]
    num_events = len(consensus_txt)

    event_timing_dir = os.path.join(args.event_timing_dir,story)
    if not os.path.exists(event_timing_dir):
        os.makedirs(event_timing_dir)
    # get onset and offset of each event
    event_timing_df = get_event_timing_df(story,idx_of_boundaries,consensus_wordlist,timing_df,num_events)
    event_timing_df.to_csv(os.path.join(event_timing_dir,'event_timing_sec.csv'),index=False)
    # split and recombine the events
    df_recombined_aligned = split_and_recombine_event(event_timing_df,story,timing_df,original_txt)
    df_recombined_aligned.to_csv(os.path.join(event_timing_dir,'event_recombined.csv'),index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--story")
    parser.add_argument("--save_dir", default = "/home/jianing/generation/generated/")
    parser.add_argument("--segmentation_dir",default = '/home/jianing/generation/behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default='/home/jianing/generation/transcripts/moth_stories')
    parser.add_argument("--timing_dir",default = '/home/jianing/generation/transcripts/timing')
    parser.add_argument("--event_timing_dir", default = '/home/jianing/generation/behavior_data/event_timing')
    args = parser.parse_args()
    main(args)