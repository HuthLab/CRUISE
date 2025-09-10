'''
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
def load_segmentation_data(path):
    '''
    load participants' segmentatino df and extract IDs 
    '''
    data = pd.read_csv(path)
    data = data[:len(data)-1]
    participant_id = data['Participant Private ID'].unique().astype(int).tolist()
    prolific_id = data['Participant Public ID'].unique().tolist()
    story = data.loc[data.index[0]]['Spreadsheet']
    if 'round2' in story:
        story = story.replace('_round2', '')
    return data,participant_id,prolific_id,story

def get_segmenation_timing(participant_id,data,story):
    '''
    Extracts segmentation timing in ms after the start of story audio
    Input:
        participant_id:private IDs from gorilla
        data: segmentation dataframes 
        story: string, story name 
    Output: 
        all_segmentation_timings: lit of lists, each sublist is one participant's segmentation
        movie_col: list of story name 
    '''
    segmentation_dict = {}
    movie_col = []
    for unique_id in participant_id:
        participant = data.loc[data['Participant Private ID']==unique_id]
        participant = participant.loc[participant['display']=='Listening Trials']
        if story == 'pieman':
            story_audio = story+'_audio.mp3'
        else:
            story_audio = story+'.mp3'
        participant = participant.loc[participant['audio']==story_audio]
        min_participant_index = min(participant.index)
        start_audio_row = np.where(np.logical_and(participant['display']=='Listening Trials',participant['Response']=='AUDIO PLAY REQUESTED'))[0][0]
        start_audio_timing = float(participant['Reaction Time'][min_participant_index+start_audio_row]) # timing in ms since start of this screen
        segmentation_timings = participant['Reaction Time'][participant['Response']=='normal.png'].astype(float)
        relative_timings = (segmentation_timings-start_audio_timing)/1000
        segmentation_dict[unique_id] = relative_timings
    return segmentation_dict,movie_col

def get_comprehension_acc(participant_id,prolific_id,comprehension_path,story,save_dir,experiment_name):
    q1,q2,q3,q4,q5 = [],[],[],[],[] # comprehension question 
    comprehension_data = pd.read_csv(comprehension_path)
    comprehension_data = comprehension_data[:len(comprehension_data)-1]
    comprehension_data['Participant Private ID'] = comprehension_data['Participant Private ID'].astype(int)
    for unique_id in participant_id:
        if unique_id ==10165622 and story =='avatar': # this subject answered on prolific afterward
            q1.append(1)
            q2.append(1)
            q3.append(0)
            q4.append(1)
            q5.append(1)
        if unique_id ==11960375 and story =='wheretheressmoke':# this subject answered on prolific afterward
            q1.append(1)
            q2.append(1)
            q3.append(1)
            q4.append(1)
            q5.append(1)
        elif unique_id not in list(comprehension_data['Participant Private ID']):
            print('subject %s is missing comprehension data for story %s' %(unique_id,story))
            q1.append(np.nan)
            q2.append(np.nan)
            q3.append(np.nan)
            q4.append(np.nan)
            q5.append(np.nan)
        else:
            participant_comprehension = comprehension_data.loc[comprehension_data['Participant Private ID']==unique_id]
            q1.append(participant_comprehension['Correct'].loc[participant_comprehension['Screen Name']=='Question 1'].values[0])
            q2.append(participant_comprehension['Correct'].loc[participant_comprehension['Screen Name']=='Question 2'].values[0])
            q3.append(participant_comprehension['Correct'].loc[participant_comprehension['Screen Name']=='Question 3'].values[0])
            q4.append(participant_comprehension['Correct'].loc[participant_comprehension['Screen Name']=='Question 4'].values[0])
            q5.append(participant_comprehension['Correct'].loc[participant_comprehension['Screen Name']=='Question 5'].values[0])
    comprehension_df = pd.DataFrame(list(zip(participant_id,prolific_id,q1,q2,q3,q4,q5)),
               columns =['subject','prolific_id','q1','q2','q3','q4','q5'])
    comprehension_df['story'] = np.repeat(story,len(comprehension_df))
    # save acc 
    comprehension_df.to_csv(os.path.join(save_dir,'comprehension_%s_%s.csv'%(story,experiment_name)),index = False)
    # print acc 
    subject_mean_acc = np.mean(comprehension_df[['q1','q2','q3','q4','q5']],axis = 1)
    question_mean_acc = np.mean(comprehension_df[['q1','q2','q3','q4','q5']],axis = 0)
    print(story)
    print(question_mean_acc)
    fig,ax = plt.subplots(figsize = (5,5))
    ax.hist(subject_mean_acc)
    ax.set_title('Subject comprehension accuracy for story %s'%story)
    fig.savefig(os.path.join(save_dir,'comprehension_acc_%s_%s.png'%(story,experiment_name)))
    poor_comp_subjects = comprehension_df['subject'][np.where(subject_mean_acc<0.6)[0]].astype(int)
    return comprehension_df,poor_comp_subjects

def segmentation_timing_to_word_ind(story:str,segmentation_dict:dict,participant_id:List[int],prolific_id:List[str],timing_dir:str,exclude:List[int],poor_comp_subjects:List[int],save_path:str,experiment_name):
    '''
    convert segmentation timing in sec to the index of the closest upcoming word in text after the segmentation
    returns: 
        all_closest_starts: idx of the closest upcoming word in the transcript
        segmentation_count: number of segmentation of each subject 
        good_participant_id: good participants ids (private ie gorilla id) after exclusion
        good_prolific_id: good participant prolific ie public id after exclusion
    '''
    story_timings = pd.read_csv(os.path.join(timing_dir,'%s_timing.csv'%story))

    # align segmentation times to words
    start_times = story_timings['start'].values
    all_closest_starts = [] # idx of the closest upcoming word 
    segmentation_count = []
    good_participant_id = []
    good_prolific_id = []
    for i,sub in tqdm(enumerate(participant_id)):
        if sub not in segmentation_dict.keys():
            print('%d did not segment'%participant_id)
            continue
        if len(exclude)!=0 and sub in exclude:
            continue
        if sub in poor_comp_subjects:
            continue
        closest_starts = []
        subject_segmentation = segmentation_dict[sub]
        if len(subject_segmentation)==0:
            print('subject %d did not segment'%sub)
            exclude.append(sub)
            all_closest_starts.append(np.array([]))
            segmentation_count.append(0)
            good_participant_id.append(sub)
            good_prolific_id.append(prolific_id[i])
            continue
        for button in subject_segmentation:
            closest_start = np.where(button-start_times<=0)[0] 
            #print('button',button,'closest_start ind',closest_start[0], 'closest start val',start_times[closest_start[0]])
            if len(closest_start) != 0:
                closest_start = closest_start[0] # the first word that starts after the button press
                closest_starts.append(closest_start)
    #         else: 
    #             if button > max(start_times):
    #                 print('subject %d pressed button after story ends'%sub)
        all_closest_starts.append(np.array(closest_starts))
        segmentation_count.append(len(closest_starts))
        if len(closest_starts) < 3:
            print('subject %d segmented %d times for story %s'%(sub,len(closest_starts),story))
        good_participant_id.append(sub)
        good_prolific_id.append(prolific_id[i])
    print(story)
    print('mean segmentation count',np.mean(segmentation_count),'median ',np.median(segmentation_count))
    print('min',np.min(segmentation_count),'max ',np.max(segmentation_count))
    print('good participants count: %d'%len(good_participant_id))
    # plot histogram
    fig,ax  = plt.subplots(figsize = (5,5))
    ax.hist(segmentation_count)
    ax.set_title('Subject segmentation count for %s'%story)
    if not os.path.exists(os.path.join(save_path,story)):
        os.makedirs(os.path.join(save_path,story))
    fig.savefig(os.path.join(save_path,story,'segmentation_count_hist_%s_%s.png'%(story,experiment_name)))
    return all_closest_starts,segmentation_count,good_participant_id,good_prolific_id

def segmentation_ind_to_txt(all_closest_starts:List[np.ndarray],good_participant_id:List[int],good_prolific_id:List[str],timing_dir:str,story:str,save_path,experiment_name:str):
    '''
    Convert word indices of segmentation obtained from segmentation_timing_to_word_ind into paragraph
    using \n to denote segmentation
    Saves segmentation dataset
    '''
    original_txt = []
    segmented_txt = []
    segmented_txt_no_space = []
    movie_col = []
    subject = []

    story_timings = pd.read_csv(os.path.join(timing_dir,'%s_timing.csv'%story))
    story_text = story_timings.text.values
    story_text = [t if not pd.isnull(t) else '' for t in story_text] # sometimes a line's text is empty string. It'll be read as nan and cause issues
    joined_story_text = ' '.join(story_text)
    for i, subj_start in tqdm(enumerate(all_closest_starts)):
        # insert new lines at the start of the chunk_segment_idx
        chunk_segmented = story_text.copy()
        if len(all_closest_starts)==0:
            no_space_txt = story_text.copy()
        else:
            for start_idx in subj_start:
                if '\n' not in chunk_segmented[start_idx]:# if the participant haven't already pressed the button for this word
                    chunk_segmented[start_idx] = '\n' + chunk_segmented[start_idx]
            chunk_segmented_txt = ' '.join(chunk_segmented)
            
            # get rid of space before \n
            split_txt = chunk_segmented_txt.split('\n')
            split_txt = [s for s in split_txt if s != '']
            split_txt_no_space = [s[:-1] if s[-1] == ' ' else s for s in split_txt]
            no_space_txt = '\n'.join(split_txt_no_space)

        # update 
        original_txt.append(joined_story_text) # append to the list of original text 
        segmented_txt.append(chunk_segmented_txt)
        movie_col.append(story)
        subject.append(i)
        segmented_txt_no_space.append(no_space_txt)
    if not os.path.exists(os.path.join(save_path,story)):
        os.makedirs(os.path.join(save_path,story))
    story_df = pd.DataFrame(list(zip(movie_col, good_participant_id,good_prolific_id,original_txt, segmented_txt,segmented_txt_no_space)),
                    columns =['story', 'subject','prolific_id','original_txt','segmented_txt','segmented_txt_no_space'])
    story_df.to_csv(os.path.join(save_path,story,'%s_segmentation_%s.csv'%(story,experiment_name)),index = False)
    return story_df

def segmentation_to_word_list(human_output):
    '''
    Input: list of strings, each string is an event 
    Output: list of strings, each string is a word. The string has a \n at the end of the word if the human segmented after that point 
    '''
    human_output_newline = [s[:-1] if s[-1] == ' ' else s for s in human_output] # takes care of whitespace
    human_output_newline = [s[1:] if s[0] == ' ' else s for s in human_output_newline] # takes care of whitespace
    human_output_newline = [s+'\n' for s in human_output_newline] # add \n to end of each segmentation 
    human_output_split = ' '.join(human_output_newline).split(' ')
    return human_output_split

def calculate_consensus(story,story_df,consensus_save_dir,experiment_name):
    # get proportion of segmentation at the end of each segmentation boundary
    # first check that human segmented txt have same number of words as original
    story_transcript = story_df['original_txt'][0]
    for i in story_df.index:
        if len(story_df['segmented_txt'][i].split(' '))!=len(story_df['original_txt'][i].split(' ')):
            print(i)
    human_segmentation_proportion = np.zeros(len(story_transcript.split(' ')))
    num_subjects = len(story_df)
    for i in story_df.index:
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
    with open(os.path.join(consensus_save_dir,story,'%s_consensus_%s.txt'%(story,experiment_name)),'w') as f:
        f.write(calculated_consensus_txt)
    with open(os.path.join(consensus_save_dir,story,'%s_consensus_wordlist_%s.pkl'%(story,experiment_name)),'wb') as f:
        pickle.dump(consensus_wordlist,f)

    # plot segmentation
    fig,ax = plt.subplots(figsize = (10,5))
    ax.plot(np.arange(len(segmentation_prop_smoothed)),segmentation_prop_smoothed,color = 'blue',label = 'smoothed')
    ax.plot(np.arange(len(human_segmentation_proportion)),human_segmentation_proportion, color = 'red',alpha = 0.5,label = 'original')
    ax.hlines(threshold,xmin = 0, xmax = len(segmentation_prop_smoothed))
    ax.set_title('segmentation data for %s: N = %d'%(story,num_subjects))
    ax.plot(peaks, segmentation_prop_smoothed[peaks], "x",color = 'black',label = 'Consensus')
    ax.legend()
    fig.savefig(os.path.join(consensus_save_dir,story,'%s_segmentation_prop_%s.png'%(story,experiment_name)))
    with open(os.path.join(consensus_save_dir,story,'%s_segmentation_prop.pkl'%story),'wb') as f:
        pickle.dump(human_segmentation_proportion,f)

def main(args):
    spreadsheet_paths = glob.glob(os.path.join(args.dir,'*.csv'))
    save_path = args.dir+'_parsed'
    experiment_name = os.path.split(args.dir)[-1]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nstories_per_branch = args.nstory_per_branch
    nbranches = args.nbranches
    if nbranches ==1:
        branch_a_story_1_path = [p for p in spreadsheet_paths if args.branch_a_story1 in p][0]
        branch_a_story_1_comp_path = [p for p in spreadsheet_paths if args.branch_a_story1_comp in p][0]
        branch_a_story_1_data,branch_a_story_1_private_id,branch_a_story_1_prolific_id,branch_a_story_1_name = load_segmentation_data(branch_a_story_1_path)
        branch_a_story_1_seg_timings,branch_a_story_1_col = get_segmenation_timing(branch_a_story_1_private_id,branch_a_story_1_data,branch_a_story_1_name)
        branch_a_story_1_comp_df,branch_a_story_1_poor_comp_subs = get_comprehension_acc(branch_a_story_1_private_id,branch_a_story_1_prolific_id,branch_a_story_1_comp_path,branch_a_story_1_name,save_path,experiment_name)
        branch_a_story_1_all_closest_starts,branch_a_story_1_segmentation_count,branch_a_story_1_good_participant_id,branch_a_story_1_good_prolific_id = segmentation_timing_to_word_ind(branch_a_story_1_name,branch_a_story_1_seg_timings,branch_a_story_1_private_id,branch_a_story_1_prolific_id,args.timing_dir,[],branch_a_story_1_poor_comp_subs,args.consensus_dir,experiment_name)
        branch_a_story_1_segmentation_df = segmentation_ind_to_txt(branch_a_story_1_all_closest_starts,branch_a_story_1_good_participant_id,branch_a_story_1_good_prolific_id,args.timing_dir,branch_a_story_1_name,args.consensus_dir,experiment_name)
        calculate_consensus(branch_a_story_1_name,branch_a_story_1_segmentation_df,args.consensus_dir,experiment_name)
    else:
        # load files and parse 
        branch_a_story_1_path = [p for p in spreadsheet_paths if args.branch_a_story1 in p][0]
        branch_b_story_1_path = [p for p in spreadsheet_paths if args.branch_b_story1 in p][0]
        if nstories_per_branch >1:
            branch_a_story_2_path = [p for p in spreadsheet_paths if args.branch_a_story2 in p][0]
            branch_b_story_2_path = [p for p in spreadsheet_paths if args.branch_b_story2 in p][0]

        # comprehension files 
        branch_a_story_1_comp_path = [p for p in spreadsheet_paths if args.branch_a_story1_comp in p][0]
        branch_b_story_1_comp_path = [p for p in spreadsheet_paths if args.branch_b_story1_comp in p][0]
        if nstories_per_branch >1:
            branch_a_story_2_comp_path = [p for p in spreadsheet_paths if args.branch_a_story2_comp in p][0]
            branch_b_story_2_comp_path = [p for p in spreadsheet_paths if args.branch_b_story2_comp in p][0]
        
        # load segmentation dfs
        branch_a_story_1_data,branch_a_story_1_private_id,branch_a_story_1_prolific_id,branch_a_story_1_name = load_segmentation_data(branch_a_story_1_path)
        branch_b_story_1_data,branch_b_story_1_private_id,branch_b_story_1_prolific_id,branch_b_story_1_name =load_segmentation_data(branch_b_story_1_path)
        if nstories_per_branch >1:
            branch_a_story_2_data,branch_a_story_2_private_id,branch_a_story_2_prolific_id,branch_a_story_2_name = load_segmentation_data(branch_a_story_2_path)
            branch_b_story_2_data,branch_b_story_2_private_id,branch_b_story_2_prolific_id,branch_b_story_2_name = load_segmentation_data(branch_b_story_2_path)

        # segmentation timings
        branch_a_story_1_seg_timings,branch_a_story_1_col = get_segmenation_timing(branch_a_story_1_private_id,branch_a_story_1_data,branch_a_story_1_name)
        branch_b_story_1_seg_timings,branch_b_story_1_col = get_segmenation_timing(branch_b_story_1_private_id,branch_b_story_1_data,branch_b_story_1_name)
        if nstories_per_branch >1:
            branch_a_story_2_seg_timings,branch_a_story_2_col = get_segmenation_timing(branch_a_story_2_private_id,branch_a_story_2_data,branch_a_story_2_name)
            branch_b_story_2_seg_timings,branch_b_story_2_col = get_segmenation_timing(branch_b_story_2_private_id,branch_b_story_2_data,branch_b_story_2_name)
        
        # comprehension 
        branch_a_story_1_comp_df,branch_a_story_1_poor_comp_subs = get_comprehension_acc(branch_a_story_1_private_id,branch_a_story_1_prolific_id,branch_a_story_1_comp_path,branch_a_story_1_name,save_path,experiment_name)
        branch_b_story_1_comp_df,branch_b_story_1_poor_comp_subs = get_comprehension_acc(branch_b_story_1_private_id, branch_b_story_1_prolific_id, branch_b_story_1_comp_path, branch_b_story_1_name,save_path,experiment_name)
        if nstories_per_branch >1:
            branch_a_story_2_comp_df,branch_a_story_2_poor_comp_subs = get_comprehension_acc(branch_a_story_2_private_id, branch_a_story_2_prolific_id, branch_a_story_2_comp_path, branch_a_story_2_name,save_path,experiment_name)
            branch_b_story_2_comp_df,branch_b_story_2_poor_comp_subs = get_comprehension_acc(branch_b_story_2_private_id, branch_b_story_2_prolific_id, branch_b_story_2_comp_path, branch_b_story_2_name,save_path,experiment_name)
        
        # convert segmentation timings in sec to word indices in text 
        branch_a_story_1_all_closest_starts,branch_a_story_1_segmentation_count,branch_a_story_1_good_participant_id,branch_a_story_1_good_prolific_id = segmentation_timing_to_word_ind(branch_a_story_1_name,branch_a_story_1_seg_timings,branch_a_story_1_private_id,branch_a_story_1_prolific_id,args.timing_dir,[],branch_a_story_1_poor_comp_subs,args.consensus_dir,experiment_name)
        branch_b_story_1_all_closest_starts, branch_b_story_1_segmentation_count, branch_b_story_1_good_participant_id, branch_b_story_1_good_prolific_id = segmentation_timing_to_word_ind(branch_b_story_1_name, branch_b_story_1_seg_timings, branch_b_story_1_private_id, branch_b_story_1_prolific_id, args.timing_dir, [], branch_b_story_1_poor_comp_subs,args.consensus_dir,experiment_name)
        if nstories_per_branch >1:
            branch_a_story_2_all_closest_starts, branch_a_story_2_segmentation_count, branch_a_story_2_good_participant_id, branch_a_story_2_good_prolific_id = segmentation_timing_to_word_ind(branch_a_story_2_name, branch_a_story_2_seg_timings, branch_a_story_2_private_id, branch_a_story_2_prolific_id, args.timing_dir, [], branch_a_story_2_poor_comp_subs,args.consensus_dir,experiment_name)
            branch_b_story_2_all_closest_starts, branch_b_story_2_segmentation_count, branch_b_story_2_good_participant_id, branch_b_story_2_good_prolific_id = segmentation_timing_to_word_ind(branch_b_story_2_name, branch_b_story_2_seg_timings, branch_b_story_2_private_id, branch_b_story_2_prolific_id, args.timing_dir, [], branch_b_story_2_poor_comp_subs,args.consensus_dir,experiment_name)

        # convert segmentation word indices to paragraphs with \n denoting segmentations
        branch_a_story_1_segmentation_df = segmentation_ind_to_txt(branch_a_story_1_all_closest_starts,branch_a_story_1_good_participant_id,branch_a_story_1_good_prolific_id,args.timing_dir,branch_a_story_1_name,args.consensus_dir,experiment_name)
        branch_b_story_1_segmentation_df = segmentation_ind_to_txt(branch_b_story_1_all_closest_starts, branch_b_story_1_good_participant_id, branch_b_story_1_good_prolific_id, args.timing_dir, branch_b_story_1_name,args.consensus_dir,experiment_name)
        if nstories_per_branch >1:
            branch_a_story_2_segmentation_df = segmentation_ind_to_txt(branch_a_story_2_all_closest_starts, branch_a_story_2_good_participant_id, branch_a_story_2_good_prolific_id, args.timing_dir, branch_a_story_2_name,args.consensus_dir,experiment_name)
            branch_b_story_2_segmentation_df = segmentation_ind_to_txt(branch_b_story_2_all_closest_starts, branch_b_story_2_good_participant_id, branch_b_story_2_good_prolific_id, args.timing_dir, branch_b_story_2_name,args.consensus_dir,experiment_name)
        
        # calculate consensus and save 
        calculate_consensus(branch_a_story_1_name,branch_a_story_1_segmentation_df,args.consensus_dir,experiment_name)
        calculate_consensus(branch_b_story_1_name, branch_b_story_1_segmentation_df, args.consensus_dir,experiment_name)
        if nstories_per_branch >1:
            calculate_consensus(branch_a_story_2_name, branch_a_story_2_segmentation_df, args.consensus_dir,experiment_name)
            calculate_consensus(branch_b_story_2_name, branch_b_story_2_segmentation_df, args.consensus_dir,experiment_name)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',default = '../behavior_data/data_exp_166306-v2',help='directory where gorilla data spreadsheets are saved')
    parser.add_argument('--timing_dir',default = '../behavior_data/transcripts/timing')
    parser.add_argument('--consensus_dir',default='../behavior_data/segmentation')
    parser.add_argument('--branch_a_story1',default = 'task-bpch',help = 'audio task name of branch a, story 1')
    parser.add_argument('--branch_a_story2',default = 'task-o78z',help = 'audio task name of branch a, story 2')
    parser.add_argument('--branch_b_story1',default = 'task-ilwh',help = 'audio task name of branch b, story 1')
    parser.add_argument('--branch_b_story2',default = 'task-az4e',help = 'audio task name of branch b, story 2')
    parser.add_argument('--branch_a_story1_comp',default = 'task-ukpf',help = 'comprehension task name of branch a, story 1')
    parser.add_argument('--branch_a_story2_comp',default = 'task-4xvk',help = 'comprehension task name of branch a, story 2')
    parser.add_argument('--branch_b_story1_comp',default = 'task-8ef9',help = 'comprehension task name of branch b, story 1')
    parser.add_argument('--branch_b_story2_comp',default = 'task-a1dn',help = 'comprehension task name of branch b, story 2')
    parser.add_argument('--nstory_per_branch',default = 1,choices = [1,2],type = int, help ='number of stories per branch')
    parser.add_argument('--nbranches',type = int, default = 2)
    args = parser.parse_args()
    main(args)