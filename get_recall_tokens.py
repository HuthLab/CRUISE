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

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    story = args.story

    if args.model_recall or args.model_recall_with_entropy:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'model_recall')
    elif args.verbatim:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'verbatim_recall')
    elif args.story=='sherlock':
        save_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_truncated')
    else:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'prolific_data')
    
    if args.story=='sherlock':
        moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_truncated')
        # load story tokens
        story_tokens = torch.load(os.path.join(moth_output_dir,'tokens.pkl'))
        with open(os.path.join(moth_output_dir,'tokenized_txt.pkl'),'rb') as f:
            tokenized_txt = pickle.load(f)
    else:
        moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'moth_stories_output')
    
    system_prompt = '''You are a human with limited memory ability. You're going to listen to a story, and your task is to recall the story and summarize it in your own words in a verbal recording. Respond as if you’re speaking out loud.''' 
    
    if args.model_recall:
        #corrected_transcript = pd.read_csv(os.path.join(args.recall_transcript_dir,'%s_model_recall_transcript.csv'%story))
        if args.temp is not None:
            corrected_transcript = pd.read_csv(os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s.csv'%(story,args.temp,args.prompt_number,args.att_to_story_start)))
        else:
            corrected_transcript = pd.read_csv(os.path.join(save_dir,'%s_model_recall_transcript.csv'%story))
    elif args.model_recall_with_entropy:
        corrected_transcript = pd.read_csv(os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s_new.csv'%(story,args.temp,args.prompt_number,args.att_to_story_start)))
    elif args.verbatim:
        corrected_transcript = pd.read_csv(os.path.join(save_dir,'%s_verbatim_recall_transcripts.csv'%story))
    else:
        recall_transcript_dir = os.path.join(args.recall_transcript_dir,story) 
        corrected_transcript = pd.read_csv(os.path.join(recall_transcript_dir,'%s_corrected_recall_transcripts.csv'%story))
    corrected_transcript = corrected_transcript.dropna(axis = 0) # drop bad subjects (nan in corrected transcript)
    remove_punctuation = string.punctuation.translate(str.maketrans('', '', '\'')) # remove all punctuation except ' cuz abbreviations
    if args.recall_original_concat or args.original_recall_concat:
        if args.story =='sherlock':
            with open(os.path.join(args.sherlock_transcript_dir,'transcript_for_recall.txt'),'r') as f:
                original_txt = f.read()
            story_tokens = torch.load(os.path.join(moth_output_dir,'tokens.pkl'))
        else:
            with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
                original_txt = f.read()
            story_tokens = torch.load(os.path.join(moth_output_dir,story,'tokens.pkl'))
        if story_tokens.device.type=='cuda':
            story_tokens = story_tokens.detach().cpu()
        
    if args.recall_only:
        recall_tokens_dict = {}
        subject_ids = []
        recall_tokens = []
        for i in tqdm(corrected_transcript.index):
            subject_id = corrected_transcript['subject'][i]
            transcript = corrected_transcript['corrected transcript'][i]
            no_punctuation_transcript = transcript.translate(str.maketrans('', '', remove_punctuation))
            no_punctuation_transcript = no_punctuation_transcript.lower()
            if no_punctuation_transcript.startswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[1:]
            if no_punctuation_transcript.endswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[:-1]
            
            recall_tokenized = tokenizer(no_punctuation_transcript, return_tensors="pt").input_ids
            subject_ids.append(subject_id)
            recall_tokens.append(recall_tokenized)

        recall_tokens_dict['subject_id'] = subject_ids
        recall_tokens_dict['input_tokenized'] = recall_tokens
        
        if args.temp is not None:
            if args.model_recall_with_entropy:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}_new")
            else:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}")
        elif args.story =='sherlock':
            tokens_save_dir = save_dir
        else:
            tokens_save_dir = os.path.join(save_dir,story)
        if not os.path.isdir(tokens_save_dir):
            os.makedirs(tokens_save_dir)
        with open(os.path.join(tokens_save_dir,'recall_tokens.pkl'),'wb') as f:
            pickle.dump(recall_tokens_dict,f)

    if args.recall_original_concat:
        original_transcript_tokenized = tokenizer(original_txt, return_tensors="pt",add_special_tokens = False).input_ids
        assert torch.equal(story_tokens[0,1:],original_transcript_tokenized[0]),'existing and new tokenization of the original transcript must be the same'
        #print('story tokens shape',story_tokens.shape,original_transcript_tokenized.shape)
        recall_original_tokens_dict = {}
        subject_ids = []
        original_transcript_start_indicies = []
        all_input_tokenized = []
        max_token_len = 0

        for i in tqdm(corrected_transcript.index):
            subject_id = corrected_transcript['subject'][i]
            transcript = corrected_transcript['corrected transcript'][i]
            no_punctuation_transcript = transcript.translate(str.maketrans('', '', remove_punctuation))
            no_punctuation_transcript = no_punctuation_transcript.lower()
            if no_punctuation_transcript.startswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[1:]
            if no_punctuation_transcript.endswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[:-1]
            
            recall_tokenized = tokenizer(no_punctuation_transcript, return_tensors="pt").input_ids
            input_tokenized = torch.cat((recall_tokenized,original_transcript_tokenized),axis = 1)
            original_transcript_start_index = recall_tokenized.shape[1]
            #print('recall tokens shape:',recall_tokenized.shape)
            #print(story_tokens[0,1:].shape,input_tokenized[0,original_transcript_start_index:].shape)
            assert torch.equal(story_tokens[0,1:],input_tokenized[0,original_transcript_start_index:])
            
            subject_ids.append(subject_id)
            original_transcript_start_indicies.append(original_transcript_start_index)
            all_input_tokenized.append(input_tokenized)
            if input_tokenized.shape[1] > max_token_len:
                max_token_len = input_tokenized.shape[1]
                
        recall_original_tokens_dict['subject_id'] = subject_ids
        recall_original_tokens_dict['original_transcript_start_index'] = original_transcript_start_indicies 
        recall_original_tokens_dict['input_tokenized'] = all_input_tokenized
        
        if args.temp is not None:
            if args.model_recall_with_entropy:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}_new")
            else:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}")
        elif args.story =='sherlock':
            tokens_save_dir = save_dir
        else:
            tokens_save_dir = os.path.join(save_dir,story)
        with open(os.path.join(tokens_save_dir,'recall_original_concat.pkl'),'wb') as f:
            pickle.dump(recall_original_tokens_dict,f)
    if args.original_recall_concat:
        original_transcript_tokenized = tokenizer(original_txt, return_tensors="pt").input_ids
        assert torch.equal(story_tokens,original_transcript_tokenized),'existing and new tokenization of the original transcript must be the same'
        original_recall_tokens_dict = {}
        subject_ids = []
        recall_start_indicies = []
        all_input_tokenized = []
        if args.instruct:
            story_start_indices = []
            story_end_indices = []
        max_token_len = 0

        for i in tqdm(corrected_transcript.index):
            subject_id = corrected_transcript['subject'][i]
            transcript = corrected_transcript['corrected transcript'][i]
            no_punctuation_transcript = transcript.translate(str.maketrans('', '', remove_punctuation))
            no_punctuation_transcript = no_punctuation_transcript.lower()
            if no_punctuation_transcript.startswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[1:]
            if no_punctuation_transcript.endswith(' '):
                no_punctuation_transcript = no_punctuation_transcript[:-1]
            if args.instruct:
                user_prompt = "Here's the story: %s\nHere's your recall: "%original_txt
                messages = [
                    {"role": "system","content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "system","content": no_punctuation_transcript},
                ]
                input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
                colon_indices = []
                for i,t in enumerate(input_tokenized[0]):
                    txt = tokenizer.decode(t)
                    if ':' in txt:
                        colon_indices.append(i+1)
                assert len(colon_indices)>=2
                story_start_index = colon_indices[0]
                recall_start_index = colon_indices[1]
                len_original = original_transcript_tokenized.shape[1]
                story_end_index = story_start_index+len_original
                original_decoded = tokenizer.decode(original_transcript_tokenized[0])
                instruct_decoded = tokenizer.decode(input_tokenized[0,story_start_index:story_end_index])
                assert original_decoded.split() == instruct_decoded.split(),'original story tokenized in the two ways must be identical'
                
                story_start_indices.append(story_start_index)
                story_end_indices.append(story_end_index)
                
            else:
                recall_tokenized = tokenizer(no_punctuation_transcript, return_tensors="pt",add_special_tokens = False).input_ids
                input_tokenized = torch.cat((original_transcript_tokenized,recall_tokenized),axis = 1)
                recall_start_index = original_transcript_tokenized.shape[1]
                assert torch.equal(story_tokens[0],input_tokenized[0,:recall_start_index])
            
            subject_ids.append(subject_id)
            recall_start_indicies.append(recall_start_index)
            all_input_tokenized.append(input_tokenized)
            if input_tokenized.shape[1] > max_token_len:
                max_token_len = input_tokenized.shape[1]
                
        original_recall_tokens_dict['subject_id'] = subject_ids
        original_recall_tokens_dict['recall_start_index'] = recall_start_indicies 
        original_recall_tokens_dict['input_tokenized'] = all_input_tokenized
        if args.instruct:
            original_recall_tokens_dict['story_start_index'] = story_start_indices
            original_recall_tokens_dict['story_end_index'] = story_end_indices
        print('max token len:',max_token_len)
        if args.temp is not None:
            if args.model_recall_with_entropy:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}_new")
            else:
                tokens_save_dir = os.path.join(save_dir,f"{story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}")
        elif args.story =='sherlock':
            tokens_save_dir = save_dir
        else:
            tokens_save_dir = os.path.join(save_dir,story)
        if args.instruct:
            tokens_save_dir = tokens_save_dir+'_instruct'
        if not os.path.exists(tokens_save_dir):
            os.makedirs(tokens_save_dir)
        with open(os.path.join(tokens_save_dir,'original_recall_concat.pkl'),'wb') as f:
            pickle.dump(original_recall_tokens_dict,f)
    if args.story_concat:
        stories = ['pieman','alternateithicatom', 'avatar', 'howtodraw', 'legacy', 
            'life', 'myfirstdaywiththeyankees', 'naked', 
            'odetostepfather', 'souls', 'undertheinfluence',
            'stagefright', 'tildeath', 'sloth', 'exorcism', 'haveyoumethimyet', 
           'adollshouse', 'inamoment', 'theclosetthatateeverything', 'adventuresinsayingyes',
           'buck', 'swimmingwithastronauts', 'thatthingonmyarm', 'eyespy', 'itsabox', 'hangtime',
           'fromboyhoodtofatherhood',
           'wheretheressmoke']
        all_input_tokenized= []
        original_transcript_start_indicies = []
        original_story_concat_dict = {}

        for story in tqdm(stories):
            story_tokens = torch.load(os.path.join(moth_output_dir,story,'tokens.pkl'))
            story_tokens_nobos = story_tokens[:,1:]
            input_tokenized = torch.cat((story_tokens,story_tokens_nobos),axis = 1)
            original_transcript_start_index = story_tokens.shape[1]
            #print(story_tokens.shape,input_tokenized.shape)
            all_input_tokenized.append(input_tokenized)
            original_transcript_start_indicies.append(original_transcript_start_index)
            
        original_story_concat_dict['story'] = stories
        original_story_concat_dict['original_transcript_start_index'] = original_transcript_start_indicies
        original_story_concat_dict['input_tokenized'] = all_input_tokenized
        original_concat_save_dir = os.path.join(save_dir,'original_transcript_concat')
        if not os.path.exists(original_concat_save_dir):
            os.makedirs(original_concat_save_dir)
        save_path = os.path.join(original_concat_save_dir,'original_concat.pkl')
        with open(save_path,'wb') as f:
            pickle.dump(original_story_concat_dict,f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir", default = "/home/jianing/generation/generated/")
    parser.add_argument("--recall_transcript_dir",default = '/home/jianing/generation/behavior_data/recall_transcript')
    parser.add_argument("--original_transcript_dir",default='/home/jianing/generation/transcripts/moth_stories')
    parser.add_argument("--moth_output_dir",default = '/home/jianing/generation/generated/')
    parser.add_argument("--recall_only",action='store_true')
    parser.add_argument("--recall_original_concat",action = 'store_true')
    parser.add_argument("--original_recall_concat",action = 'store_true')
    parser.add_argument("--story_concat",action='store_true')
    parser.add_argument("--story",default = 'pieman',help = 'to run the concatenated entropy of original stories, enter original')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--model_recall",action='store_true')
    parser.add_argument("--temp",type = float,help = 'temperature to set for model generation')
    parser.add_argument("--att_to_story_start",action ='store_true',help = 'limit the modified attention to the start of story, not the start of sys prompt')
    parser.add_argument("--prompt_number",type = int,default = 0,help = 'prompt number')
    parser.add_argument("--instruct",help = 'use instruct prompt')
    parser.add_argument("--verbatim",action='store_true',help = 'verbatim recall experiment')
    parser.add_argument("--sherlock_transcript_dir",default = '/home/jianing/generation/sherlock')
    parser.add_argument("--model_recall_with_entropy",action = 'store_true',help = 'new model recalls with entropy computed')
    args = parser.parse_args()
    main(args)