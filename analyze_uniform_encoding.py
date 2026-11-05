'''generate the uniform_encoding_df.csv, works for both even split windows and events'''
import torch
import os
import pandas as pd
import numpy as np
import pickle
from utils import get_segmentation_indices,segmentation_to_word_list
from utils import model_to_path_dict
import argparse

def get_pairwise_event_explained_ce(pairwise_event_save_dir,calibration_temp):
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_tokens.pkl'),'rb') as f:
        token_pairs_dict = pickle.load(f)
    with open(os.path.join(pairwise_event_save_dir,'pairwise_event_startidx_of_second_event.pkl'),'rb') as f:
        startidx_of_second_event_dict = pickle.load(f)
    if calibration_temp is not None:
        with open(os.path.join(pairwise_event_save_dir,'pairwise_event_ce_caltemp_%.2f.pkl'%calibration_temp),'rb') as f:
            pairwise_event_ce_dict = pickle.load(f)
        with open(os.path.join(pairwise_event_save_dir,'event_only_ce_caltemp_%.2f.pkl'%calibration_temp),'rb') as f:
            all_events_only_ce = pickle.load(f)
    else:
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

def get_weighted_window_val(bin_starts, bin_ends, val, win_start, win_end):
    overlaps = np.maximum(0, np.minimum(bin_ends, win_end) - np.maximum(bin_starts, win_start))
    bin_dur = bin_ends - bin_starts

    weights = overlaps / bin_dur
    weighted_sum = np.sum(val * weights)

    return weighted_sum

def get_sylber_results(save_dir,story,sr_exp,shrink):
    '''
    Get token level sylber results
    sr_exp, shrink: bool
    '''
    sylber_save_dir_name = model_to_path_dict['Sylber']['save_dir_name']
    sylber_save_dir = os.path.join(save_dir,sylber_save_dir_name)

    if not sr_exp:
        condition ='original'
    else:
        if shrink:
            condition = 'shrink'
        else:
            condition = 'original_nobackground'
    sylber_story_results_dir = os.path.join(sylber_save_dir,condition,story)
    with open(os.path.join(sylber_story_results_dir,'segments.npy'),'rb') as f:
        segments = np.load(f)
    with open(os.path.join(sylber_story_results_dir,'cross_entropy.pkl'),'rb') as f:
        ce = pickle.load(f)
    ce = ce.detach().cpu().numpy()
    assert segments.shape[0] == ce.shape[0]
    return segments,ce,condition

def get_sylber_results_by_window(save_dir,story,sr_exp,shrink,story_split_by_duration_df):
    fr = 50 # Hz, Sylber frame rate is 50 Hz, so each segment corresponds to 20ms.
    segments,ce,condition = get_sylber_results(save_dir,story,sr_exp,shrink) # token level results
    segments_sec = segments*(1/fr)
    sylber_window_ce = []
    for i,row in story_split_by_duration_df.iterrows():
        row_onset = row['Onset']
        row_offset = row['Offset']
        weighted_sum = get_weighted_window_val(segments_sec[:,0], segments_sec[:,1], ce, row_onset, row_offset)
        sylber_window_ce.append(weighted_sum)
    return sylber_window_ce

def exponential_saturation_tau(tau,duration,ceiling):
    '''standard exponential saturation, where rate of saturation is determined by tau (time constant) and ceiling'''
    return ceiling*(1-np.exp(-duration/tau))

def get_saturation_model(taus,event_ceiling,parts_duration,pairwise_event_explained_ce_dict,all_events_only_ce,event_len):
    exp_duration_dict = {tau:[] for tau in taus}
    exp_duration_sum_dict = {tau:[] for tau in taus}
    for tau in taus:
        exp_duration = exponential_saturation_tau(tau,parts_duration,event_ceiling)
        all_events_sum = get_duration_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict,exp_duration)
        exp_duration_dict[tau] = exp_duration
        exp_duration_sum_dict[tau] = all_events_sum
    return exp_duration_dict,exp_duration_sum_dict

def align_offset(event_timing_df):
    aligned_offset = []
    for i,offset in enumerate(event_timing_df['Offset']):
        if i==len(event_timing_df['Offset'])-1:
            aligned_offset.append(offset)
        else:
            if offset<event_timing_df['Onset'].iloc[i+1]:
                aligned_offset.append(event_timing_df['Onset'].iloc[i+1])
            else:
                aligned_offset.append(offset)
    aligned_offset = np.array(aligned_offset)
    return np.array(aligned_offset)

def main(args):
    story = args.story
    adjusted = args.adjusted
    save_dir = args.save_dir
    model_name = args.model
    calibration_temp = args.calibration_temp
    factor = args.factor
    sr_save_dir = args.sr_save_dir
    model_save_dir_name = model_to_path_dict[model_name]['save_dir_name']
    model_initial_char = model_to_path_dict[model_name]['initial_char']
    moth_output_dir = os.path.join(save_dir,model_save_dir_name,'moth_stories_output')
    # tokenized txt
    with open(os.path.join(moth_output_dir,story,'tokenized_txt.pkl'),'rb') as f:
        tokenized_txt = pickle.load(f)
    if calibration_temp is not None:
        with open(os.path.join(moth_output_dir,story,'cross_entropy_caltemp_%.2f.pkl'%calibration_temp),'rb') as f:
            original_ce = pickle.load(f)
        kl = np.load(os.path.join(moth_output_dir,story,'kl_caltemp_%.2f.npy'%calibration_temp)) # this includes the predictive distribution after the sequence ends
    else:
        with open(os.path.join(moth_output_dir,story,'cross_entropy.pkl'),'rb') as f:
            original_ce = pickle.load(f)
        kl = np.load(os.path.join(moth_output_dir,story,'kl.npy')) # this includes the predictive distribution after the sequence ends
    kl = kl[:-1] # originally the last one is word t-1 predicting t and t predicting nonexistent next word. This mismatches with the surprisal calculation, which does not account for the last word predicting the next. So now this is from token position 1 to the end of story
    mean_kl = np.mean(kl) # pad KL for 0th token in the story with mean
    kl = np.insert(kl,0,mean_kl)

    consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%story)
    with open(consensus_path,'r') as f:
        consensus_txt = f.read()
    consensus_txt = consensus_txt.split('\n')
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
        original_txt = f.read()

    if args.sr_exp:
        participant_info_df = pd.read_csv(os.path.join(args.exclusion_dir,'participant_info','%s_participant_info.csv'%story))
        if args.shrink:
            exclusion_df = pd.read_csv(os.path.join(args.exclusion_dir,'shrink','%s_exclusion.csv'%story))
            participants = participant_info_df.loc[participant_info_df['condition']=='shrink']['gorilla_id']
        else:
            exclusion_df = pd.read_csv(os.path.join(args.exclusion_dir,'original','%s_exclusion.csv'%story))
            participants = participant_info_df.loc[participant_info_df['condition']=='original']['gorilla_id']
    else:
        exclusion_df = pd.read_csv(os.path.join(args.exclusion_dir,'%s_exclusion.csv'%story))
    excluded_subjects = exclusion_df['excluded'].loc[exclusion_df['recall']==1].values

    if args.event:
        consensus_wordlist = segmentation_to_word_list(consensus_txt)
        segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char=model_initial_char)
        event_len = np.diff([0]+segmentation_indices_in_tokens)
        event_len[0]+=1
        pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story)
        sr_recall_output_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'pairwise_event',story)

        conditional_event_ce = []
        event_kl = []
        for count,idx in enumerate(segmentation_indices_in_tokens):
            if count ==0:
                start_idx = 0
            else:
                start_idx = segmentation_indices_in_tokens[count-1]
            end_idx = idx+1
            conditional_event_ce.append(torch.sum(original_ce[start_idx:end_idx]))
            event_kl.append(np.sum(kl[start_idx:end_idx]))

        event_timing_dir = os.path.join(args.event_timing_dir,story)
        if args.sr_exp and args.shrink:
            event_timing_df = pd.read_csv(os.path.join(event_timing_dir,'event_timing_sec_shrink_-%dto%d_cosine.csv'%(args.shrink_window,args.shrink_window)))
            original_event_timing_df = pd.read_csv(os.path.join(event_timing_dir,'event_timing_sec.csv'))
        else:
            event_timing_df = pd.read_csv(os.path.join(event_timing_dir,'event_timing_sec.csv'))
            original_event_timing_df = event_timing_df.copy()

        if args.align_offset:# align offset to onset of the next event
            aligned_offset = align_offset(event_timing_df)
            event_duration = aligned_offset - event_timing_df['Onset']
            original_aligned_offset = align_offset(original_event_timing_df)
            original_event_duration = original_aligned_offset - original_event_timing_df['Onset']
        else:
            event_duration = event_timing_df['Offset'] - event_timing_df['Onset']
            original_event_duration = original_event_timing_df['Offset'] - original_event_timing_df['Onset']
    else:
        if args.split_by_token:
            if adjusted:
                if args.factor:
                    save_dir_name = 'story_split_tokens_factor_%.1f_adjusted'%factor
                    pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                    sr_recall_output_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                    if args.sr_exp and args.shrink:
                        shrink_story_split_df_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name+'_shrunk')
                        story_split_by_duration_df = pd.read_csv(os.path.join(shrink_story_split_df_save_dir,'story_split_by_token_df_adjusted.csv'))
                    else:
                        story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df_adjusted.csv'))
                    original_story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_token_df_adjusted.csv'))
        else:
            # there are a few directories here.
            # 1. pairwise_event_save_dir: the directory of the original experiment, contains the full pairwise event inference results and split df for the original stories
            # 2. sr_recall_output_save_dir: the directory of the sr_exp experiment, contains the recall mi for both conditions of sr_exp
            # 3. shrink_story_split_df_save_dir: the directory that contains the split df for the shrunk stories in sr_exp
            # Across exps, pairwise story window mi are read from pairwise_event_save_dir. recall mi is read from the sr_recall_output_save_dir for both conditions of sr_exp
            # split_df is read from shrink_story_split_df_save_dir for sr_exp shrunk condition, and from pairwise_event_save_dir for original condition and original exp.
            if adjusted:
                if args.factor:
                    save_dir_name = 'story_split_timing_factor_%.1f_adjusted'%factor
                    pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                    sr_recall_output_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                else:
                    save_dir_name = 'story_split_timing_adjusted'
                    pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                    sr_recall_output_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                if args.sr_exp and args.shrink:
                    shrink_story_split_df_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name+'_shrunk')
                    story_split_by_duration_df = pd.read_csv(os.path.join(shrink_story_split_df_save_dir,'story_split_by_duration_df_adjusted.csv'))
                else:
                    story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df_adjusted.csv'))
                original_story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df_adjusted.csv'))
            else:
                save_dir_name = 'story_split_timing'
                pairwise_event_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                sr_recall_output_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name)
                if args.sr_exp and args.shrink:
                    shrink_story_split_df_save_dir = os.path.join(save_dir,model_save_dir_name,'pairwise_event',story,save_dir_name+'_shrunk')
                    story_split_by_duration_df = pd.read_csv(os.path.join(shrink_story_split_df_save_dir,'story_split_by_duration_df.csv'))
                else:
                    story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'))
                original_story_split_by_duration_df = pd.read_csv(os.path.join(pairwise_event_save_dir,'story_split_by_duration_df.csv'))
        event_len=story_split_by_duration_df['event_ends'] - story_split_by_duration_df['event_starts']
        event_duration = story_split_by_duration_df['Duration']
        original_event_duration = original_story_split_by_duration_df['Duration']

        # conditional CE
        conditional_event_ce = []
        event_kl = []
        for idx in range(len(story_split_by_duration_df)):
            start_idx = story_split_by_duration_df['event_starts'].iloc[idx]
            end_idx = story_split_by_duration_df['event_ends'].iloc[idx]
            conditional_event_ce.append(torch.sum(original_ce[start_idx:end_idx]))
            event_kl.append(np.sum(kl[start_idx:end_idx]))

    conditional_event_ce = np.array(conditional_event_ce)
    event_kl = np.array(event_kl)
    if args.event:
        window_timing_df = event_timing_df
    else:
        window_timing_df = story_split_by_duration_df
    sylber_ce = get_sylber_results_by_window(save_dir,story,args.sr_exp,args.shrink,window_timing_df)
    pairwise_event_explained_ce_dict = get_pairwise_event_explained_ce(pairwise_event_save_dir,calibration_temp)

    if calibration_temp is not None:
        recall_first_df_name = 'recall_explained_event_ce_df_caltemp_%.2f.csv'%calibration_temp
        recall_last_df_name = 'event_explained_recall_ce_df_caltemp_%.2f.csv'%calibration_temp
        with open(os.path.join(pairwise_event_save_dir,'event_only_ce_caltemp_%.2f.pkl'%calibration_temp),'rb') as f:
            all_events_only_ce = pickle.load(f)
    else:
        recall_first_df_name = 'recall_explained_event_ce_df.csv'
        recall_last_df_name = 'event_explained_recall_ce_df.csv'
        with open(os.path.join(pairwise_event_save_dir,'event_only_ce.pkl'),'rb') as f:
            all_events_only_ce = pickle.load(f)

    if args.sr_exp:
        recall_output_save_dir = sr_recall_output_save_dir
        uniform_encoding_save_dir = os.path.join(sr_save_dir,model_save_dir_name,'uniform_encoding',story)
    else:
        recall_output_save_dir = pairwise_event_save_dir
        uniform_encoding_save_dir = os.path.join(save_dir,model_save_dir_name,'uniform_encoding',story)

    # recall data
    # direct concatenation
    if args.random_recalls:
        recall_explained_event_info_df = pd.read_csv(os.path.join(recall_output_save_dir,'random_recalls',recall_first_df_name))
    else:
        recall_explained_event_info_df = pd.read_csv(os.path.join(recall_output_save_dir,recall_first_df_name))
        if args.sr_exp: # for sr exp, need to select the participants in that condition, then exclude
            recall_explained_event_info_df = recall_explained_event_info_df[recall_explained_event_info_df['subject'].isin(participants)]
        recall_explained_event_info_df = recall_explained_event_info_df[~recall_explained_event_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect = recall_explained_event_info_df.groupby('event')['ER_intersect'].mean()

    if args.random_recalls:
        event_explained_recall_info_df_raw = pd.read_csv(os.path.join(recall_output_save_dir,'random_recalls',recall_last_df_name))
    else:
        event_explained_recall_info_df_raw = pd.read_csv(os.path.join(recall_output_save_dir,recall_last_df_name))
        if args.sr_exp:
            event_explained_recall_info_df_raw = event_explained_recall_info_df_raw[event_explained_recall_info_df_raw['subject'].isin(participants)]
        event_explained_recall_info_df_raw = event_explained_recall_info_df_raw[~event_explained_recall_info_df_raw['subject'].isin(excluded_subjects)]
    mean_ER_intersect_recall_last = event_explained_recall_info_df_raw.groupby('event')['ER_intersect'].mean()

    # instruct
    if args.random_recalls:
        recall_explained_info_df = pd.read_csv(os.path.join(recall_output_save_dir,'random_recalls','instruct',recall_first_df_name))
    else:
        recall_explained_info_df = pd.read_csv(os.path.join(recall_output_save_dir,'instruct',recall_first_df_name))
        if args.sr_exp:
            recall_explained_info_df = recall_explained_info_df[recall_explained_info_df['subject'].isin(participants)]
        recall_explained_info_df = recall_explained_info_df[~recall_explained_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect_instruct = recall_explained_info_df.groupby('event')['ER_intersect'].mean()

    if args.random_recalls:
        event_explained_recall_info_df = pd.read_csv(os.path.join(recall_output_save_dir,'random_recalls','instruct',recall_last_df_name))
    else:
        event_explained_recall_info_df = pd.read_csv(os.path.join(recall_output_save_dir,'instruct',recall_last_df_name))
        if args.sr_exp:
            event_explained_recall_info_df = event_explained_recall_info_df[event_explained_recall_info_df['subject'].isin(participants)]
        event_explained_recall_info_df = event_explained_recall_info_df[~event_explained_recall_info_df['subject'].isin(excluded_subjects)]
    mean_ER_intersect_recall_last_instruct = event_explained_recall_info_df.groupby('event')['ER_intersect'].mean()

    if args.factor is not None:
        assert len(story_split_by_duration_df)==len(mean_ER_intersect)==len(mean_ER_intersect_recall_last)==len(mean_ER_intersect_instruct)==len(mean_ER_intersect_recall_last_instruct)
    else:
        assert len(consensus_txt)==len(mean_ER_intersect)==len(mean_ER_intersect_recall_last)==len(mean_ER_intersect_instruct)==len(mean_ER_intersect_recall_last_instruct)

    # uniform encoding, conditional
    all_events_sum_conditional = get_conditional_uniform_encoding(event_len,all_events_only_ce,conditional_event_ce,pairwise_event_explained_ce_dict)
    all_events_sum = get_event_len_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict)
    all_events_sum_by_duration = get_duration_uniform_encoding(event_len,all_events_only_ce,pairwise_event_explained_ce_dict,event_duration)
    all_events_sum_kl = get_conditional_uniform_encoding(event_len,all_events_only_ce,event_kl,pairwise_event_explained_ce_dict)
    all_events_sum_sylber_ce = get_conditional_uniform_encoding(event_len,all_events_only_ce,sylber_ce,pairwise_event_explained_ce_dict)
    all_events_only_sum_ce = np.array([torch.sum(t) for t in all_events_only_ce])

    # saturation model with different tau values
    taus = np.linspace(2.5,10,7)
    exp_duration_saturation_dict, exp_duration_saturation_sum_dict = get_saturation_model(taus,original_event_duration,event_duration,pairwise_event_explained_ce_dict,all_events_only_ce,event_len)
    saturation_model_dict= {'exponential_samples':exp_duration_saturation_dict,'exponential_uniform_encoding':exp_duration_saturation_sum_dict}

    if not os.path.exists(uniform_encoding_save_dir):
        os.makedirs(uniform_encoding_save_dir)
    uniform_encoding_df = pd.DataFrame({'H(event|prev events)':conditional_event_ce,
                                        'H(event)':all_events_only_sum_ce,
                                        'event_len':event_len,
                                        'event_duration':event_duration,
                                        'event_kl':event_kl,
                                        "sylber_ce":sylber_ce,
                                        'weighted_event_info':all_events_sum,
                                        'weighted_event_info_by_duration':all_events_sum_by_duration,
                                        'weighted_event_info_conditioned':all_events_sum_conditional,
                                        'weighted_event_info_kl':all_events_sum_kl,
                                        "weighted_event_info_sylber_ce":all_events_sum_sylber_ce,
                                        'mean_ER_intersect':mean_ER_intersect,
                                        'mean_ER_intersect_recall_last':mean_ER_intersect_recall_last,
                                        'mean_ER_intersect_instruct':mean_ER_intersect_instruct,
                                        'mean_ER_intersect_instruct_recall_last':mean_ER_intersect_recall_last_instruct
                            })
    if args.event:
        if args.align_offset:
            uniform_encoding_save_name = 'uniform_encoding_df_align_offset'
        else:
            uniform_encoding_save_name = 'uniform_encoding_df'
    else:
        if adjusted:
            if args.random_recalls:
                uniform_encoding_save_name = 'uniform_encoding_df_even_split_adjusted_random_recalls'
            elif args.split_by_token:
                if args.factor:
                    uniform_encoding_save_name = 'uniform_encoding_df_even_split_by_tokens_factor_%.1f_adjusted'%factor
            else: # split by duration
                if args.factor:
                    uniform_encoding_save_name = 'uniform_encoding_df_even_split_factor_%.1f_adjusted'%factor
                else:
                    uniform_encoding_save_name = 'uniform_encoding_df_even_split_adjusted'
        else:
            uniform_encoding_save_name = 'uniform_encoding_df_even_split'
    if calibration_temp is not None:
        uniform_encoding_save_name += '_caltemp_%.2f'%calibration_temp
    if args.sr_exp:
        if args.shrink:
            uniform_encoding_save_name += '_sr_shrink'
        else:
            uniform_encoding_save_name += '_sr_original'
    uniform_encoding_df.to_csv(os.path.join(uniform_encoding_save_dir,uniform_encoding_save_name+'.csv'),index = False)

    duration_capacity_model_save_name = uniform_encoding_save_name.replace('uniform_encoding_df', 'duration_exponential_saturation')
    with open(os.path.join(uniform_encoding_save_dir,duration_capacity_model_save_name+'.pkl'),'wb') as f:
        pickle.dump(saturation_model_dict,f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",default='../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--exclusion_dir",default = "../behavior_data/exclusion",help = "directory storing exclusion dfs")
    parser.add_argument("--event_timing_dir",default = '../behavior_data/event_timing',help = "directory storing event timing dfs")
    parser.add_argument("--story",default = 'pieman')
    parser.add_argument("--model")
    parser.add_argument("--adjusted",action = 'store_true',help = 'whether to use adjusted even split')
    parser.add_argument("--random_recalls",action = 'store_true',help = 'whether to use randomly sampled recalls from other stories')
    parser.add_argument("--event",action = 'store_true',help ='use events instead of equal duration windows')
    parser.add_argument("--align_offset",action ='store_true',help='align end of event to start of next event')
    parser.add_argument("--split_by_token",action = 'store_true',help = 'Split story into even number of tokens')
    parser.add_argument("--factor",type = float,help = "split into factor * number of events")
    parser.add_argument("--calibration_temp",type = float, help = 'calibration temperature')
    parser.add_argument("--sr_save_dir", default = "../exp2/generated")
    parser.add_argument("--sr_exp", action = 'store_true',help = 'use recalls from speech rate exp')
    parser.add_argument("--shrink",action = 'store_true',help = 'for sr_exp: shrink condition. If false, use original condition in sr_exp')
    parser.add_argument("--shrink_window", default = 10, type = float, help = 'shrink -/+ this many ms around the silences')
    args = parser.parse_args()
    main(args)
