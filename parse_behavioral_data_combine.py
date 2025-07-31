'''
Combine behavioral data of the same story from multiple experiments
Get segmentation timing of each participant, saves all segmentation into {story}_segmentation.csv
calculate consensus segmentation, comprehension acc
'''
import numpy as np
import pandas as pd
import os
import re 
import argparse
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import pickle

def calculate_consensus2(story,story_df,consensus_save_dir,excluded_subjects):
    # get proportion of segmentation at the end of each segmentation boundary
    # first check that human segmented txt have same number of words as original
    story_transcript = story_df['original_txt'][0]
    for i in story_df.index:
        if len(story_df['segmented_txt'][i].split(' '))!=len(story_df['original_txt'][i].split(' ')):
            print(i)
    human_segmentation_proportion = np.zeros(len(story_transcript.split(' ')))
    num_subjects = len(story_df)
    segmentation_count = []
    for i in story_df.index:
        subject_segment_count = len(story_df['segmented_txt'][i].split('\n'))
        segmentation_count.append(subject_segment_count)
        if subject_segment_count<=1: # this subject didn't segment
            num_subjects-=1
            continue
        human_txt_split = story_df['segmented_txt'][i].split(' ')
        newline_idx = [i-1 for i,word in enumerate(human_txt_split) if '\n' in word]
        if -1 in newline_idx: # some subjects segment before the the story starts 
            newline_idx.remove(-1) 
        human_segmentation_proportion[newline_idx]+=1
    human_segmentation_proportion = human_segmentation_proportion/num_subjects
    segmentation_prop_smoothed = gaussian_filter1d(human_segmentation_proportion,2.5)
    threshold = np.percentile(segmentation_prop_smoothed,95)
    segmentation_above_threshold = segmentation_prop_smoothed>=threshold
    print(story)
    print('num above 95% percentile:',sum(segmentation_above_threshold))
    ## find peaks 
    peaks, _ = find_peaks(segmentation_prop_smoothed, height=threshold)
    print('num peaks:',len(peaks))

    transcript_wordlist = story_transcript.split(' ')
    consensus_wordlist = [w+'\n' if i in peaks else w for i,w in enumerate(transcript_wordlist)]
    calculated_consensus_txt = ' '.join(consensus_wordlist)
    if excluded_subjects is None:
        with open(os.path.join(consensus_save_dir,story,'%s_consensus_pre-exclude.txt'%(story)),'w') as f:
            f.write(calculated_consensus_txt)
        with open(os.path.join(consensus_save_dir,story,'%s_consensus_wordlist_pre-exclude.pkl'%(story)),'wb') as f:
            pickle.dump(consensus_wordlist,f)
    else:
        with open(os.path.join(consensus_save_dir,story,'%s_consensus.txt'%(story)),'w') as f:
            f.write(calculated_consensus_txt)
        with open(os.path.join(consensus_save_dir,story,'%s_consensus_wordlist.pkl'%(story)),'wb') as f:
            pickle.dump(consensus_wordlist,f)

    # plot segmentation
    fig,ax = plt.subplots(figsize = (10,5))
    ax.plot(np.arange(len(segmentation_prop_smoothed)),segmentation_prop_smoothed,color = 'blue',label = 'smoothed')
    ax.plot(np.arange(len(human_segmentation_proportion)),human_segmentation_proportion, color = 'red',alpha = 0.5,label = 'original')
    ax.hlines(threshold,xmin = 0, xmax = len(segmentation_prop_smoothed))
    ax.set_title('segmentation data for %s: N = %d'%(story,num_subjects))
    ax.plot(peaks, segmentation_prop_smoothed[peaks], "x",color = 'black',label = 'Consensus')
    ax.legend()
    if excluded_subjects is None:
        fig.savefig(os.path.join(consensus_save_dir,story,'%s_segmentation_prop_pre-exclude.png'%(story)))
        with open(os.path.join(consensus_save_dir,story,'%s_segmentation_prop_pre-exclude.pkl'%story),'wb') as f:
            pickle.dump(human_segmentation_proportion,f)
    else:
        fig.savefig(os.path.join(consensus_save_dir,story,'%s_segmentation_prop.png'%(story)))
        with open(os.path.join(consensus_save_dir,story,'%s_segmentation_prop.pkl'%story),'wb') as f:
            pickle.dump(human_segmentation_proportion,f)
    # plot segmentation count histogram
    fig,ax  = plt.subplots(figsize = (5,5))
    ax.hist(segmentation_count)
    ax.set_title('Subject segmentation count for %s'%story)
    assert len(segmentation_count) == len(story_df)
    segmentation_count_df = pd.DataFrame(list(zip(story_df['story'],story_df['subject'],story_df['prolific_id'],segmentation_count)),
                                         columns =['story', 'subject','prolific_id','segmentation_count'])
    if excluded_subjects is None:
        fig.savefig(os.path.join(consensus_save_dir,story,'segmentation_count_hist_%s_pre-exclude.png'%(story)))
        segmentation_count_df.to_csv(os.path.join(consensus_save_dir,story,'segmentation_count_%s_pre-exclude.csv'%story),index = False)
    else:
        fig.savefig(os.path.join(consensus_save_dir,story,'segmentation_count_hist_%s.png'%(story)))
        segmentation_count_df.to_csv(os.path.join(consensus_save_dir,story,'segmentation_count_%s.csv'%story),index = False)
    
def main(args):
    story = args.story
    excluded_subjects = None
    if args.exclude:
        exclusion_dir = os.path.join(args.parent_dir,'exclusion')
        exclusion_df = pd.read_csv(os.path.join(exclusion_dir,'%s_exclusion.csv'%story))
        # subjects excluded for segmentation analysis
        excluded_subjects = exclusion_df['excluded'].loc[exclusion_df['segmentation']==1].values 
    # comprehension 
    save_dir = os.path.join(args.parent_dir,'comprehension',story)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp1_save_path = os.path.join(args.parent_dir,args.exp1+'_parsed')
    exp1_comprehension_df = pd.read_csv(os.path.join(exp1_save_path,'comprehension_%s_%s.csv'%(story,args.exp1)))
    # segmentation
    exp1_storydf = pd.read_csv(os.path.join(args.consensus_dir,story,'%s_segmentation_%s.csv'%(story,args.exp1)))
    if args.exp2!='none':
        exp2_save_path = os.path.join(args.parent_dir,args.exp2+'_parsed')
        exp2_comprehension_df = pd.read_csv(os.path.join(exp2_save_path,'comprehension_%s_%s.csv'%(story,args.exp2)))
        # joined df
        comprehension_df = pd.concat([exp1_comprehension_df, exp2_comprehension_df], join="inner")
        exp2_storydf = pd.read_csv(os.path.join(args.consensus_dir,story,'%s_segmentation_%s.csv'%(story,args.exp2)))
        story_df = pd.concat([exp1_storydf,exp2_storydf],join = 'inner',ignore_index = True)
    else:
        comprehension_df = exp1_comprehension_df
        story_df = exp1_storydf

    if excluded_subjects is None:
        comprehension_df.to_csv(os.path.join(save_dir,'comprehension_%s_pre-exclude.csv'%(story)),index = False)
    else:
        comprehension_df = comprehension_df[~comprehension_df['subject'].isin(excluded_subjects)]
        comprehension_df.to_csv(os.path.join(save_dir,'comprehension_%s.csv'%(story)),index = False)
    subject_mean_acc = np.mean(comprehension_df[['q1','q2','q3','q4','q5']],axis = 1)
    question_mean_acc = np.mean(comprehension_df[['q1','q2','q3','q4','q5']],axis = 0)
    print(story)
    print(question_mean_acc)
    fig,ax = plt.subplots(figsize = (5,5))
    ax.hist(subject_mean_acc)
    ax.set_title('Subject comprehension accuracy for story %s'%story)
    if excluded_subjects is None:
        fig.savefig(os.path.join(save_dir,'comprehension_acc_%s_pre-exclude.png'%(story)))
    else:
        fig.savefig(os.path.join(save_dir,'comprehension_acc_%s.png'%(story)))


    if excluded_subjects is not None:
        story_df = story_df[~story_df['subject'].isin(excluded_subjects)]
    calculate_consensus2(story,story_df,args.consensus_dir,excluded_subjects)
    if excluded_subjects is None:
        story_df.to_csv(os.path.join(args.consensus_dir,story,'%s_segmentation_pre-exclude.csv'%(story)),index = False)
    else:
        story_df.to_csv(os.path.join(args.consensus_dir,story,'%s_segmentation.csv'%(story)),index = False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir',default = '/home/jianing/generation/behavior_data/')
    parser.add_argument('--exp1',default = 'data_exp_166306-v2',help='experiment1')
    parser.add_argument('--exp2',default = 'data_exp_156140-v6',help='experiment2, if this story only has 1 exp, type none')
    parser.add_argument('--story',default = 'alternateithicatom',help = 'story to merge across experiments')
    parser.add_argument('--timing_dir',default = '/home/jianing/generation/transcripts/timing')
    parser.add_argument('--consensus_dir',default='/home/jianing/generation/behavior_data/segmentation')
    parser.add_argument('--exclude', action ='store_true',help = 'calculate segmentation while excluding certain subjects')
    args = parser.parse_args()
    main(args)