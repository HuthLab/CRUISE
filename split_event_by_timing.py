import pandas as pd
import numpy as np


def align_timing_with_text(df_recombined,timing_df,original_txt,story):
    '''
    realign the timing to text, because the actual onset and offset of the adjusted events 
    will be different from the original division
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