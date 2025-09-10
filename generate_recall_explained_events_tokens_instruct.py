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
from utils import model_to_path_dict

def get_event_tokens_instruct_base_model(tokenizer,event_txt,recall = False,subjects = None):
    '''For base model. Same instruct prompt, but no chat template'''
    event_tokens_instruct =[]
    event_start_indices = []
    event_end_indices = []
    no_prompt_tokens = [] # no prompt, just the recall or event tokens without bos

    if recall:
        prompt = '''You are going to read a human's recall of a story. Here's the recall: '''
    else:
        prompt = '''You are going to read a segment from a story. Here's the segment from the story: '''
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids # 1 * n tokens, has bos
    for event_transcript in tqdm(event_txt):
        while event_transcript.startswith(' '):
            event_transcript = event_transcript[1:]
        if event_transcript.endswith(' '):
            event_transcript = event_transcript[:-1]
        event_tokens = tokenizer(event_transcript, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
        concat_tokens = torch.cat([prompt_tokens,event_tokens],dim=1)
        event_start_indices.append(prompt_tokens.shape[1])
        event_end_indices.append(concat_tokens.shape[1])
        event_tokens_instruct.append(concat_tokens)
        no_prompt_tokens.append(event_tokens)
    if recall:
        event_stimuli_instruct = {'input_tokens':event_tokens_instruct,
                        'recall_start_indices':event_start_indices,
                        'recall_end_indices':event_end_indices,
                        'subject':subjects,
                        'no_prompt_tokens':no_prompt_tokens}
    else:
        event_stimuli_instruct = {'input_tokens':event_tokens_instruct,
                                'event_start_indices':event_start_indices,
                                'event_end_indices':event_end_indices,
                                'no_prompt_tokens':no_prompt_tokens}
    return event_stimuli_instruct

def get_event_recall_tokens_instruct_base_model(tokenizer,event_txt,recall_instruct,corrected_transcript,subjects):
    '''For base model. Same instruct prompt, but no chat template'''
    event_recall_stim_dict = {}
    system_prompt = '''You are going to read a segment from a story, along with a human's recall of the entire story this segment belongs to.'''
    story_prompt = "Here's the segment from the story: "
    recall_prompt = ". Here's the recall of the story: "
    system_prompt_tokens = tokenizer(system_prompt, return_tensors="pt").input_ids # 1 * n tokens, has bos
    story_prompt_tokens = tokenizer(story_prompt, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
    first_half_prompt_tokens = torch.cat([system_prompt_tokens,story_prompt_tokens],dim=1)
    recall_prompt_tokens = tokenizer(recall_prompt, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos

    recall_only_tokens = recall_instruct['no_prompt_tokens']
    for sub_idx,(recall,subject) in enumerate(zip(corrected_transcript,subjects)):
        while recall.startswith(' '):
            recall = recall[1:]
        if recall.endswith(' '):
            recall = recall[:-1]
        recall_tokens = tokenizer(recall, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
        assert torch.equal(recall_tokens,recall_only_tokens[sub_idx]),'recall tokens tokenized alone and here must be the same'
        
        input_tokens_instruct = []
        event_start_indices = []
        event_end_indices = []
        recall_start_indices_in_event_instruct = []
        recall_end_indices_in_event_instruct = []

        for event_idx,event_transcript in enumerate(tqdm(event_txt)):
            while event_transcript.startswith(' '):
                event_transcript = event_transcript[1:]
            if event_transcript.endswith(' '):
                event_transcript = event_transcript[:-1]
            event_tokens = tokenizer(event_transcript, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
            concat_tokens = torch.cat([first_half_prompt_tokens,event_tokens,recall_prompt_tokens,recall_tokens],dim=1)
            event_start_idx = first_half_prompt_tokens.shape[1]
            event_end_idx = first_half_prompt_tokens.shape[1]+event_tokens.shape[1]
            recall_start_idx = event_end_idx+recall_prompt_tokens.shape[1]
            recall_end_idx = recall_start_idx+recall_tokens.shape[1]
            assert recall_end_idx==concat_tokens.shape[1],'recall end idx must be the end of the token sequence'
            
            recall_start_indices_in_event_instruct.append(recall_start_idx)
            recall_end_indices_in_event_instruct.append(recall_end_idx)
            input_tokens_instruct.append(concat_tokens)
            event_start_indices.append(event_start_idx)
            event_end_indices.append(event_end_idx)
        event_recall_stim_dict[sub_idx] = {'input_tokens':input_tokens_instruct,
                                    'event_start_indices':event_start_indices,
                                    'event_end_indices':event_end_indices,
                                    'recall_start_indices':recall_start_indices_in_event_instruct,
                                    'recall_end_indices':recall_end_indices_in_event_instruct,
                                    'subject':subject
                                    }
    return event_recall_stim_dict

def get_recall_event_tokens_instruct_base_model(tokenizer,event_txt,event_stimuli_instruct,corrected_transcript):
    recall_event_stim_dict = {}
    system_prompt = '''You are going to read a human's recall of a story, and segment from a story.'''
    recall_prompt = '''Here's the recall of the story: '''
    story_prompt = '''. Here's the segment from the story: '''
    system_prompt_tokens = tokenizer(system_prompt, return_tensors="pt").input_ids # 1 * n tokens, has bos
    story_prompt_tokens = tokenizer(story_prompt, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
    recall_prompt_tokens = tokenizer(recall_prompt, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
    first_half_prompt_tokens = torch.cat([system_prompt_tokens,recall_prompt_tokens],dim=1)

    event_only_tokens = event_stimuli_instruct['no_prompt_tokens']
    for event_idx,event_transcript in enumerate(tqdm(event_txt)):
        while event_transcript.startswith(' '):
            event_transcript = event_transcript[1:]
        if event_transcript.endswith(' '):
            event_transcript = event_transcript[:-1]
        event_tokens = tokenizer(event_transcript, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
        assert torch.equal(event_tokens,event_only_tokens[event_idx]),'event tokenized alone must be the same as event tokenized here'

        input_tokens_instruct = []
        recall_start_indices = []
        recall_end_indices = []
        event_start_indices_in_recall_instruct = []
        event_end_indices_in_recall_instruct = []
        for sub,recall in enumerate(corrected_transcript):
            while recall.startswith(' '):
                recall = recall[1:]
            if recall.endswith(' '):
                recall = recall[:-1]
            recall_tokens = tokenizer(recall, return_tensors="pt",add_special_tokens= False).input_ids # 1 * n tokens, no bos
            concat_tokens = torch.cat([first_half_prompt_tokens,recall_tokens,story_prompt_tokens,event_tokens],dim=1)
            recall_start_idx = first_half_prompt_tokens.shape[1]
            recall_end_idx = recall_start_idx+recall_tokens.shape[1]
            event_start_idx = recall_end_idx+story_prompt_tokens.shape[1]
            event_end_idx = event_start_idx+event_tokens.shape[1]
            assert event_end_idx==concat_tokens.shape[1],'end of event tokens must be end of token sequence'
            
            recall_start_indices.append(recall_start_idx)
            recall_end_indices.append(recall_end_idx)
            event_start_indices_in_recall_instruct.append(event_start_idx)
            event_end_indices_in_recall_instruct.append(event_end_idx)
            input_tokens_instruct.append(concat_tokens)
            
        recall_event_stim_dict[event_idx] = {'input_tokens':input_tokens_instruct,
                                    'event_start_indices':event_start_indices_in_recall_instruct,
                                    'event_end_indices':event_end_indices_in_recall_instruct,
                                    'recall_start_indices':recall_start_indices,
                                    'recall_end_indices':recall_end_indices
                                    }
    return recall_event_stim_dict

def get_event_tokens_instruct(model_name,tokenizer,event_txt,recall = False,subjects = None):
    event_tokens_instruct =[]
    event_start_indices = []
    event_end_indices = []
    for event_transcript in tqdm(event_txt):
        while event_transcript.startswith(' '):
            event_transcript = event_transcript[1:]
        if event_transcript.endswith(' '):
            event_transcript = event_transcript[:-1]
        if recall:
            system_prompt = '''You are going to read a human's recall of a story.'''
            user_prompt = "Here's the recall: %s"%event_transcript
        else:
            system_prompt = '''You are going to read a segment from a story.'''
            user_prompt = "Here's the segment from the story: %s"%event_transcript
        if model_name =='gemma-2-9b-it':
            messages = [
                {"role": "user", "content": system_prompt+' '+user_prompt},
            ]
        else:
            messages = [
                {"role": "system","content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        colon_indices = []
        for i,t in enumerate(input_tokenized[0]):
            txt = tokenizer.decode(t)
            if ':' in txt:
                colon_indices.append(i+1)
        if 'Llama3.2-3b-instruct' in args.model: # because llama3.2 has a chat template that contains colons
            assert len(colon_indices)>=3
            story_start_index = colon_indices[2]
        else:
            assert len(colon_indices)>=1
            story_start_index = colon_indices[0]
        original_transcript_tokenized = tokenizer(event_transcript, return_tensors="pt",add_special_tokens = False).input_ids
        len_original = original_transcript_tokenized.shape[1]
        story_end_index = story_start_index+len_original 
        original_decoded = tokenizer.decode(original_transcript_tokenized[0])
        instruct_decoded = tokenizer.decode(input_tokenized[0,story_start_index:story_end_index])
        if len(original_decoded.split()) != len(instruct_decoded.split()):
            print(len(original_decoded.split()),len(instruct_decoded.split()))
            print('event transcript:',event_transcript)
            print('original decoded:',original_decoded)
            print('instruct decoded:',instruct_decoded)
            print('entire instruct prompt:',tokenizer.decode(input_tokenized[0]))
            print('before first colon:',tokenizer.decode(input_tokenized[0][:story_start_index]))
        assert len(original_decoded.split()) == len(instruct_decoded.split()),'original story tokenized in the two ways must be identical'
        event_tokens_instruct.append(input_tokenized)
        event_start_indices.append(story_start_index)
        event_end_indices.append(story_end_index)
    if recall:
        event_stimuli_instruct = {'input_tokens':event_tokens_instruct,
                        'recall_start_indices':event_start_indices,
                        'recall_end_indices':event_end_indices,
                        'subject':subjects}
    else:
        event_stimuli_instruct = {'input_tokens':event_tokens_instruct,
                                'event_start_indices':event_start_indices,
                                'event_end_indices':event_end_indices}
    return event_stimuli_instruct

def get_event_recall_tokens_instruct(model_name,tokenizer,event_txt,recall_instruct,corrected_transcript):
    event_recall_stim_dict = {}
    recall_tokens_instruct = recall_instruct['input_tokens']
    recall_start_indices = recall_instruct['recall_start_indices']
    recall_end_indices = recall_instruct['recall_end_indices']
    subjects = recall_instruct['subject']
    for sub_idx,(recall,subject) in enumerate(zip(corrected_transcript,subjects)):
        recall_tokens = recall_tokens_instruct[sub_idx][0,recall_start_indices[sub_idx]:recall_end_indices[sub_idx]]
        recall_only_decoded = tokenizer.decode(recall_tokens)
        input_tokens_instruct = []
        event_start_indices = []
        event_end_indices = []
        recall_start_indices_in_event_instruct = []
        recall_end_indices_in_event_instruct = []
        for event_idx,event_transcript in enumerate(tqdm(event_txt)):
            if recall.startswith(' '):
                recall = recall[1:]
            if recall.endswith(' '):
                recall = recall[:-1]
            if recall.startswith(' '):
                recall = recall[1:]
            if event_transcript.startswith(' '):
                event_transcript = event_transcript[1:]
            system_prompt = '''You are going to read a segment from a story, along with a human's recall of the entire story this segment belongs to.'''
            user_prompt = "Here's the segment from the story: %s. Here's the recall of the story: %s"%(event_transcript,recall)
            if model_name =='gemma-2-9b-it':
                messages = [
                    {"role": "user", "content": system_prompt+' '+user_prompt},
                ]
            else:
                messages = [
                    {"role": "system","content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
            if ':' in event_transcript:
                i = 1
                decoded = ''
                while i < input_tokenized.shape[1] and "Here's the segment from the story:" not in decoded:
                    i+=1
                    decoded = tokenizer.decode(input_tokenized[0,:i])
                assert i<input_tokenized.shape[1]
                story_start_index = i
                
                while i < input_tokenized.shape[1] and "Here's the recall of the story:" not in decoded:
                    i+=1
                    decoded = tokenizer.decode(input_tokenized[0,:i])
                assert i<input_tokenized.shape[1]
                recall_start_index = i
            else:
                colon_indices = []
                for i,t in enumerate(input_tokenized[0]):
                    txt = tokenizer.decode(t)
                    if ':' in txt: # fix the colon, make it judge based on "Here's the recall of the story:"
                        colon_indices.append(i+1)
                if 'Llama3.2-3b-instruct' in args.model:
                    assert len(colon_indices)>=4
                    story_start_index = colon_indices[2]
                    recall_start_index = colon_indices[3]
                else:
                    assert len(colon_indices)>=2
                    story_start_index = colon_indices[0]
                    recall_start_index = colon_indices[1]

            event_tokenized = tokenizer(event_transcript, return_tensors="pt",add_special_tokens = False).input_ids
            original_recall_tokenized = tokenizer(recall, return_tensors="pt",add_special_tokens = False).input_ids
            len_event = event_tokenized.shape[1]
            len_original_recall = original_recall_tokenized.shape[1]
            recall_end_index = recall_start_index+len_original_recall
            story_end_index = story_start_index+len_event 
            
            original_recall_decoded = tokenizer.decode(original_recall_tokenized[0])
            recall_instruct_decoded = tokenizer.decode(input_tokenized[0,recall_start_index:recall_end_index])
            if recall_tokens.shape != input_tokenized[0,recall_start_index:recall_end_index].shape:
                print('sub',sub_idx,'event',event_idx)
                print('recall raw',original_recall_tokenized[0].shape,'recall post instruct',recall_tokens.shape,'recall post concat',input_tokenized[0,recall_start_index:recall_end_index].shape)
                print('recall post instruct')
                print(recall_only_decoded)
                print('----------------------')
                print('recall post concat',recall_instruct_decoded)
                print('------------------------')
                print('recall alone',original_recall_decoded)
            assert recall_tokens.shape == input_tokenized[0,recall_start_index:recall_end_index].shape
            #instruct_decoded = tokenizer.decode(input_tokenized[0,story_start_index:story_end_index])
            if len(recall_only_decoded.split()) != len(recall_instruct_decoded.split()):
                print('event',event_idx,'subject',subject)
                print(len(recall_only_decoded.split()), len(recall_instruct_decoded.split()))
                #print(recall_instruct_decoded.split()[-1])
                print('----------------------')
                print(recall_instruct_decoded)
                print('----------------------')
                print(tokenizer.decode(input_tokenized[0]))
                # for i in range(len(recall_instruct_decoded.split())):
                #     print(i,recall_only_decoded.split()[i],recall_instruct_decoded.split()[i])
            assert len(recall_only_decoded.split()) == len(recall_instruct_decoded.split()),'recall tokenized in the two instruct ways must be identical'
            assert len(recall_instruct_decoded.split()) == len(original_recall_decoded.split()),'recall tokenized in the instruct vs. non-instruct must be identical'
            recall_start_indices_in_event_instruct.append(recall_start_index)
            recall_end_indices_in_event_instruct.append(recall_end_index)
            input_tokens_instruct.append(input_tokenized)
            event_start_indices.append(story_start_index)
            event_end_indices.append(story_end_index)
        event_recall_stim_dict[sub_idx] = {'input_tokens':input_tokens_instruct,
                                    'event_start_indices':event_start_indices,
                                    'event_end_indices':event_end_indices,
                                    'recall_start_indices':recall_start_indices_in_event_instruct,
                                    'recall_end_indices':recall_end_indices_in_event_instruct,
                                    'subject':subject
                                    }
    return event_recall_stim_dict
def get_recall_event_tokens_instruct(model_name,tokenizer,event_txt,event_stimuli_instruct,corrected_transcript):
    #remove_punctuation = string.punctuation.translate(str.maketrans('', '', '\'')) # remove all punctuation except ' cuz abbreviations
    recall_event_stim_dict = {}
    event_tokens_instruct = event_stimuli_instruct['input_tokens']
    event_start_indices = event_stimuli_instruct['event_start_indices']
    event_end_indices = event_stimuli_instruct['event_end_indices']
    for event_idx,event_transcript in enumerate(tqdm(event_txt)):
        event_tokens = event_tokens_instruct[event_idx][0,event_start_indices[event_idx]:event_end_indices[event_idx]]
        event_only_decoded = tokenizer.decode(event_tokens)
        input_tokens_instruct = []
        recall_start_indices = []
        recall_end_indices = []
        event_start_indices_in_recall_instruct = []
        event_end_indices_in_recall_instruct = []
        for sub,recall in enumerate(corrected_transcript):
            #recall = recall.translate(str.maketrans('', '', remove_punctuation))
            while event_transcript.startswith(' '):
                event_transcript = event_transcript[1:]
            if event_transcript.endswith(' '):
                event_transcript = event_transcript[:-1]
            if recall.startswith(' '):
                recall = recall[1:]
            # if recall.endswith(' ') or recall.endswith('\n'):
            #     recall = recall[:-1]
            system_prompt = '''You are going to read a human's recall of a story, and segment from a story.'''
            user_prompt = "Here's the recall of the story: %s. Here's the segment from the story: %s"%(recall,event_transcript)
            if model_name =='gemma-2-9b-it':
                messages = [
                    {"role": "user", "content": system_prompt+' '+user_prompt},
                ]
            else:
                messages = [
                    {"role": "system","content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
            colon_indices = []
            for i,t in enumerate(input_tokenized[0]):
                txt = tokenizer.decode(t)
                if ':' in txt:
                    colon_indices.append(i+1)
            if 'Llama3.2-3b-instruct' in args.model:
                assert len(colon_indices)>=4
                recall_start_index = colon_indices[2]
                story_start_index = colon_indices[3]
            else:
                assert len(colon_indices)>=2
                recall_start_index = colon_indices[0]
                story_start_index = colon_indices[1]
            
            recall_tokenized = tokenizer(recall, return_tensors="pt",add_special_tokens = False).input_ids
            original_transcript_tokenized = tokenizer(event_transcript, return_tensors="pt",add_special_tokens = False).input_ids
            len_recall = recall_tokenized.shape[1]
            len_original = original_transcript_tokenized.shape[1]
            recall_end_index = recall_start_index+len_recall
            story_end_index = story_start_index+len_original 
            
            recall_decoded = tokenizer.decode(recall_tokenized[0])
            original_decoded = tokenizer.decode(original_transcript_tokenized[0])
            recall_instruct_decoded = tokenizer.decode(input_tokenized[0,recall_start_index:recall_end_index])
            instruct_decoded = tokenizer.decode(input_tokenized[0,story_start_index:story_end_index])
            # technically don't need to check the alignment of recall, because you only need the indices of story to be the same
            # if len(recall_decoded.split()) != len(recall_instruct_decoded.split()):
            #     print(recall_decoded)
            #     print(recall_instruct_decoded)
            #     print(recall_decoded.split())
            #     print(recall_instruct_decoded.split())
            #     for i in range(len(recall_decoded.split())):
            #         if recall_decoded.split()[i] != recall_instruct_decoded.split()[i]:
            #             print(i,recall_decoded.split()[i],recall_instruct_decoded.split()[i])
            # assert len(recall_decoded.split()) == len(recall_instruct_decoded.split()),'recall tokenized in the two ways must be identical'
            assert len(original_decoded.split()) == len(instruct_decoded.split()),'original story tokenized in the two ways must be identical'
            assert len(event_only_decoded.split()) == len(instruct_decoded.split()),'original story tokenized in the two ways must be identical'

            event_start_indices_in_recall_instruct.append(story_start_index)
            event_end_indices_in_recall_instruct.append(story_end_index)
            input_tokens_instruct.append(input_tokenized)
            
            recall_start_indices.append(recall_start_index)
            recall_end_indices.append(recall_end_index)
        recall_event_stim_dict[event_idx] = {'input_tokens':input_tokens_instruct,
                                    'event_start_indices':event_start_indices_in_recall_instruct,
                                    'event_end_indices':event_end_indices_in_recall_instruct,
                                    'recall_start_indices':recall_start_indices,
                                    'recall_end_indices':recall_end_indices
                                    }
    return recall_event_stim_dict

def main(args):
    story = args.story
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])

    # load recall transcripts 
    recall_transcript_dir = os.path.join(args.recall_transcript_dir,story)
    if args.random_recalls:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_random_recall_transcripts.csv'%story))
    else:
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_corrected_recall_transcripts.csv'%story))
    subjects = corrected_transcript['subject'].values
    corrected_transcript = corrected_transcript.dropna(axis = 0) # drop bad subjects (nan in corrected transcript)
    corrected_transcript = corrected_transcript['corrected transcript'].values
    print('num recalls',len(corrected_transcript))
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
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing_factor_%.1f'%args.factor)
            else:
                even_split_save_dir = os.path.join(pairwise_event_save_dir,'story_split_timing')
            even_duration_split_df = pd.read_csv(os.path.join(even_split_save_dir,'story_split_by_duration_df.csv'))
        event_txt =  even_duration_split_df['text'].values
        if args.random_recalls:
            save_dir = os.path.join(even_split_save_dir,'random_recalls','instruct')
        else:
            save_dir = os.path.join(even_split_save_dir,'instruct')
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
        event_txt =  even_token_split_df['text'].values
        if args.random_recalls:
            save_dir = os.path.join(even_split_save_dir,'random_recalls','instruct')
        else:
            save_dir = os.path.join(even_split_save_dir,'instruct')
    else: # original condition
        # load consensus
        consensus_path = os.path.join(args.segmentation_dir,story,'%s_consensus.txt'%args.story)
        with open(consensus_path,'r') as f:
            consensus_txt = f.read()
        consensus_txt = consensus_txt.split('\n')
        event_txt = consensus_txt
        if args.random_recalls:
            save_dir = os.path.join(pairwise_event_save_dir,'random_recalls','instruct')
        else:
            save_dir = os.path.join(pairwise_event_save_dir,'instruct')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.recall_event_concat:
        if args.model=='Llama3-8b':
            event_stimuli_instruct = get_event_tokens_instruct_base_model(tokenizer,event_txt)
            recall_event_stim_dict = get_recall_event_tokens_instruct_base_model(tokenizer,event_txt,event_stimuli_instruct,corrected_transcript)
        else:
            event_stimuli_instruct = get_event_tokens_instruct(args.model,tokenizer,event_txt)
            recall_event_stim_dict = get_recall_event_tokens_instruct(args.model,tokenizer,event_txt,event_stimuli_instruct,corrected_transcript)
        with open(os.path.join(save_dir,'event_only_stim_instruct.pkl'),'wb') as f:
            pickle.dump(event_stimuli_instruct,f)
        with open(os.path.join(save_dir,'recall_event_stim_instruct.pkl'),'wb') as f:
            pickle.dump(recall_event_stim_dict,f)
    if args.event_recall_concat:
        if args.model=='Llama3-8b':
            recall_instruct = get_event_tokens_instruct_base_model(tokenizer,corrected_transcript,recall = True,subjects = subjects)
            event_recall_stim_dict = get_event_recall_tokens_instruct_base_model(tokenizer,event_txt,recall_instruct,corrected_transcript,subjects)
        else:
            recall_instruct = get_event_tokens_instruct(args.model,tokenizer,corrected_transcript,recall = True,subjects = subjects)
            event_recall_stim_dict = get_event_recall_tokens_instruct(args.model,tokenizer,event_txt,recall_instruct,corrected_transcript)
        with open(os.path.join(save_dir,'recall_only_stim_instruct.pkl'),'wb') as f:
            pickle.dump(recall_instruct,f)
        with open(os.path.join(save_dir,'event_recall_stim_instruct.pkl'),'wb') as f:
            pickle.dump(event_recall_stim_dict,f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--moth_output_dir",default = '../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--recall_transcript_dir",default = '../behavior_data/recall_transcript')
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