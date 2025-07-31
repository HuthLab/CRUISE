'''performs inference and calculates cross entropy and attention for each token
use flag --entropy to also cross entropy of each token
use flag --attention to calculate attention related metrics (from recall to story)
Use --model_recall to perform inference on LM generated "recall"
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
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
from utils import get_segmentation_indices,segmentation_to_word_list,model_to_path_dict
import gc

def calculate_cross_entropy(tokens,logits,base2 = False):
    # torch CE uses natural log, equivalent to -log(softmax(logit of target token))
    device = 'cuda'
    ce = F.cross_entropy(logits.to(device), tokens.to(device),reduction = 'none')
    if base2:
        ce = ce/torch.log(torch.Tensor([2])).to(device)
    ce = ce.to('cpu')
    return ce

def normalize_attn(from_to_target_attn):
    '''
    Normalize the attention matrix so the target dimension sum to 1
    This is necessary to calculate attention entropy because we're only selecting a part of the tokens
    '''
    normalized_from_to_target_attn = from_to_target_attn/np.sum(from_to_target_attn,-1,keepdims=True) 
    sum_attn_over_to = np.sum(normalized_from_to_target_attn,-1)
    ones_matrix = np.ones_like(sum_attn_over_to)
    nan_mask = np.isnan(sum_attn_over_to)
    if np.sum(nan_mask)>0:
        ones_matrix[nan_mask] = np.nan
    assert np.allclose(sum_attn_over_to,ones_matrix,atol = 1e-3,equal_nan = True)
    return normalized_from_to_target_attn

def attention_entropy(desired_token_start_idx,input_tokenized,attentions,segmentation_indices_in_tokens,include_bos = False,by_event = False):
    entropy = [] # layer * heads
    sum_attention_to_target = [] # layer * heads
    attention_by_segment = [] # layer * nevents * heads
    attention_entropy_by_segment = [] # layer *nevents * heads
    for layer_ind,layer_att in enumerate(attentions): # layer_att should be 1,num_heads, sequence_length, sequence_length
        layer_att = layer_att.detach().cpu()
        assert layer_att.shape[-1] == input_tokenized.shape[1]
        assert torch.allclose(torch.sum(layer_att,dim = -1),torch.ones_like(torch.sum(layer_att,dim = -1)),atol = 1e-3) # assuming the last dimension is the TO, it should add up to 1
        # if original-recall, want recall to original attention (ie after start idx to before start idx)
        # if recall-original, want original to recall attention (ie after start idx to before start idx)
        if not include_bos: 
            from_to_target_attn = layer_att[0,:,desired_token_start_idx:,1:desired_token_start_idx] # num_heads,from chunk length, to chunk length
        else:
            from_to_target_attn = layer_att[0,:,desired_token_start_idx:,:desired_token_start_idx] # num_heads,from chunk length, to chunk length
        from_to_target_attn = from_to_target_attn.numpy()
        normalized_attn = normalize_attn(from_to_target_attn) # normalize attn for entropy calculation
        normalized_attn = np.nan_to_num(normalized_attn)
        layer_sum_attention = np.sum(from_to_target_attn,axis = ((-1,-2))) # nheads
        assert layer_sum_attention.shape[0] == from_to_target_attn.shape[0]
        log_att = np.log2(normalized_attn) # nheads,ntoken
        log_att[log_att == np.NINF] = 0 # sometimes attn can be 0, so log is inf. Need to get rid of them otherwise results are nan
        product = np.multiply(normalized_attn,log_att) # num_heads,from chunk length, to chunk length
        layer_entropy = -np.sum(product,axis = (-1,-2)) # sum over the tokens dimension, DIM should now be num_heads
        entropy.append(layer_entropy)
        sum_attention_to_target.append(layer_sum_attention)
        layer_attention_by_segment = []
        layer_attention_entropy_by_segment = []
        if by_event:
            for i,token_ind in enumerate(segmentation_indices_in_tokens):
                if i==0:
                    segment_attention = normalized_attn[:,:,:token_ind+1] # num_heads, from (recall) chunk length, to (story event) chunk length
                    segment_unnorm_attn = from_to_target_attn[:,:,:token_ind+1]
                else:
                    segment_attention = normalized_attn[:,:,segmentation_indices_in_tokens[i-1]+1:token_ind+1]
                    segment_unnorm_attn = from_to_target_attn[:,:,segmentation_indices_in_tokens[i-1]+1:token_ind+1]
                segment_norm_attn = normalize_attn(segment_unnorm_attn)
                segment_log_att = np.log2(segment_norm_attn)
                segment_log_att[segment_log_att == np.NINF] = 0
                segment_product = np.multiply(segment_norm_attn,segment_log_att)
                assert segment_attention.shape == segment_product.shape,'shape must match'
                layer_attention_by_segment.append(np.sum(segment_attention,axis = (-1,-2)))
                segment_att_entropy = -np.sum(segment_product,axis = (-1,-2)) # sum over the tokens dimension, DIM should now be num_heads
                layer_attention_entropy_by_segment.append(segment_att_entropy)
            attention_by_segment.append(layer_attention_by_segment)
            attention_entropy_by_segment.append(layer_attention_entropy_by_segment)
        del layer_entropy,layer_sum_attention,layer_att
        gc.collect()
    del attentions
    gc.collect()
    torch.cuda.empty_cache()
    return np.array(entropy),np.array(sum_attention_to_target),np.array(attention_by_segment),np.array(attention_entropy_by_segment)


def main(args):
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    if args.attention_scale is not None:
        model = AutoModelForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],attn_implementation="eager",device_map='auto',torch_dtype = torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],device_map='auto',torch_dtype = torch.float16)
    if args.simulation:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'simulation')
    elif args.model_recall or args.model_recall_with_entropy:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'model_recall')
    elif args.attention_scale is not None:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'prolific_data')
        output_save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],f'prolific_data_modified_att_{args.attention_scale}')
    elif args.verbatim:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'verbatim_recall')
    elif args.story =='sherlock':
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_truncated')
    else:
        save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'prolific_data')
    
    if args.story =='sherlock':
        moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'sherlock_truncated')
    else:
        moth_output_dir = os.path.join(args.moth_output_dir,model_to_path_dict[args.model]['save_dir_name'],'moth_stories_output')
    
    if args.story =='original': # this files have all stories in it
        story_dir = os.path.join(save_dir,'original_transcript_concat')
        with open(os.path.join(story_dir,'original_concat.pkl'),'rb') as f:
            recall_original_tokens_dict = pickle.load(f)
    else:
        if args.model_recall and args.temp is not None:
            story_dir = os.path.join(save_dir,f"{args.story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}")
        elif args.model_recall_with_entropy and args.temp is not None:
            story_dir = os.path.join(save_dir,f"{args.story}_temp{args.temp:.2f}_prompt{args.prompt_number}_att_to_story_start_{args.att_to_story_start}_new")
        elif args.story=='sherlock':
            story_dir = save_dir
        else:
            story_dir = os.path.join(save_dir,args.story)
        with open(os.path.join(story_dir,args.file_name),'rb') as f:
            recall_original_tokens_dict = pickle.load(f)
        if args.attention_scale is not None:
            output_story_dir = os.path.join(output_save_dir,args.story)
        else:
            output_story_dir = story_dir
        if not os.path.exists(output_story_dir):
            os.makedirs(output_story_dir)
    all_input_tokenized = recall_original_tokens_dict['input_tokenized']
    print('num samples',len(all_input_tokenized))
    desired_token_start_indices = None
    if 'original_recall_concat' in args.file_name:
        desired_token_start_indices = recall_original_tokens_dict['recall_start_index']
        entropy_save_name = os.path.join(output_story_dir,'original_recall_concat_entropy.pkl')
        attention_entropy_save_name = os.path.join(output_story_dir,'original_recall_concat_attention_entropy_bos_%s.pkl'%args.include_bos)
        attention_amount_save_name = os.path.join(output_story_dir,'original_recall_concat_attention_to_original_bos_%s.pkl'%args.include_bos)
    elif 'recall_original_concat' in args.file_name:
        desired_token_start_indices = recall_original_tokens_dict['original_transcript_start_index']
        entropy_save_name = os.path.join(output_story_dir,'recall_original_concat_entropy.pkl')
        attention_entropy_save_name = os.path.join(output_story_dir,'recall_original_concat_attention_entropy_bos_%s.pkl'%args.include_bos)
        attention_amount_save_name = os.path.join(output_story_dir,'recall_original_concat_attention_to_recall_bos_%s.pkl'%args.include_bos)
    elif 'recall_tokens' in args.file_name: # just recall, no concatenation
        entropy_save_name = os.path.join(story_dir,'recall_entropy.pkl')
    
    
    if args.attention_by_event:
        with open(os.path.join(moth_output_dir,args.story,'tokenized_txt.pkl'),'rb') as f:
            tokenized_txt = pickle.load(f)
        with open(os.path.join(args.original_transcript_dir,'%s.txt'%args.story),'r') as f:
            original_txt = f.read()
        consensus_path = os.path.join(args.segmentation_dir,args.story,'%s_consensus.txt'%args.story)
        with open(consensus_path,'r') as f:
            consensus_txt = f.read()
        consensus_txt = consensus_txt.split('\n')
        consensus_wordlist = segmentation_to_word_list(consensus_txt)
        segmentation_indices_in_tokens = get_segmentation_indices(tokenized_txt,consensus_wordlist,original_txt,initial_char=model_to_path_dict[args.model]['initial_char'])
        if 'original_recall_concat' in args.file_name:
            attn_by_segment_save_name = os.path.join(output_story_dir,'original_recall_concat_attention_by_segment_bos_%s.pkl'%args.include_bos)
            attn_entropy_by_segment_save_name = os.path.join(output_story_dir,'original_recall_concat_attention_entropy_by_segment_bos_%s.pkl'%args.include_bos)
        elif 'recall_original_concat' in args.file_name:
            attn_by_segment_save_name = os.path.join(output_story_dir,'recall_original_concat_attention_by_segment_bos_%s.pkl'%args.include_bos)
            attn_entropy_by_segment_save_name = os.path.join(output_story_dir,'recall_original_concat_attention_entropy_by_segment_bos_%s.pkl'%args.include_bos)
    else:
        segmentation_indices_in_tokens = None
    
    all_entropy = []
    attn_entropy = []
    sum_attn_to_target = []
    attn_by_segment = []
    attn_entropy_by_segment = []
    exist_len = 0
    if args.entropy:
        if os.path.exists(entropy_save_name):
            with open(entropy_save_name,'rb') as f:
                all_entropy = pickle.load(f)
            exist_len = len(all_entropy)
    if args.attention:
        if os.path.exists(attention_entropy_save_name):
            with open(attention_entropy_save_name,'rb') as f:
                attn_entropy = pickle.load(f)
            with open(attention_amount_save_name,'rb') as f:
                sum_attn_to_target = pickle.load(f)
            exist_len = len(attn_entropy)
    if args.attention_by_event:
        if os.path.exists(attn_by_segment_save_name):
            with open(attn_by_segment_save_name,'rb') as f:
                attn_by_segment = pickle.load(f)
            with open(attn_entropy_by_segment_save_name,'rb') as f:
                attn_entropy_by_segment = pickle.load(f)
            exist_len = len(attn_by_segment)

    if exist_len > 0: # only evaluate new samples
        print('prev len of input:',len(all_input_tokenized))
        all_input_tokenized = all_input_tokenized[exist_len:]
        print('new len of input:',len(all_input_tokenized))
        if desired_token_start_indices is not None:
            desired_token_start_indices = desired_token_start_indices[exist_len:]

    for i,input_tokenized in tqdm(enumerate(all_input_tokenized)):
        with torch.no_grad():
            if args.attention:
                #print('Inference w/ with attention')
                #print(input_tokenized.shape)
                assert desired_token_start_indices is not None
                desired_token_start_idx = desired_token_start_indices[i]
                if args.attention_scale is not None:
                    output = model(input_tokenized.to(device),return_dict = True,output_attentions = True,recall_start_index = desired_token_start_idx,attention_scale =args.attention_scale)
                else:
                    output = model(input_tokenized.to(device),return_dict = True,output_attentions = True)
                attentions = output['attentions'] # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
                this_attn_entropy,this_sum_attn,this_attn_by_segment,this_attention_entropy_by_segment = attention_entropy(desired_token_start_idx,input_tokenized,attentions,segmentation_indices_in_tokens,include_bos=args.include_bos,by_event = args.attention_by_event)
                attn_entropy.append(this_attn_entropy)
                sum_attn_to_target.append(this_sum_attn)
                if args.attention_by_event:
                    attn_by_segment.append(this_attn_by_segment)
                    attn_entropy_by_segment.append(this_attention_entropy_by_segment)
                if args.entropy:
                    logits = output['logits']
                    # calculate cross entropy
                    entropy = calculate_cross_entropy(input_tokenized[0,1:],logits[0,:-1],base2=True) # excluding bos token 
                    all_entropy.append(entropy)
                del output,attentions
                gc.collect()
                torch.cuda.empty_cache()
            else:
                #print('Inference w/ cross entropy without attention')
                if args.attention_scale is not None:
                    assert desired_token_start_indices is not None
                    desired_token_start_idx = desired_token_start_indices[i]
                    output = model(input_tokenized.to(device),return_dict = True,recall_start_index = desired_token_start_idx,attention_scale =args.attention_scale)
                else:
                    output = model(input_tokenized.to(device),return_dict = True)
                logits = output['logits']
                # calculate cross entropy
                entropy = calculate_cross_entropy(input_tokenized[0,1:],logits[0,:-1],base2=True) # excluding bos token 
                all_entropy.append(entropy)
                del output, logits
                gc.collect()
                torch.cuda.empty_cache()
    if args.entropy:
        print(len(all_entropy),'should be equal to the total length of input')
    elif args.attention:
        print(len(attn_entropy),'should be equal to the total length of input')
    if args.story =='original':
        with open(os.path.join(story_dir,'original_concat_entropy.pkl'),'wb') as f:
            pickle.dump(all_entropy,f)
    else:
        if args.entropy:
            with open(entropy_save_name,'wb') as f:
                pickle.dump(all_entropy,f)
        if args.attention:
            with open(attention_entropy_save_name,'wb') as f:
                pickle.dump(attn_entropy,f)
            with open(attention_amount_save_name,'wb') as f:
                pickle.dump(sum_attn_to_target,f)
            if args.attention_by_event:
                with open(attn_by_segment_save_name,'wb') as f:
                    pickle.dump(attn_by_segment,f)
                with open(attn_entropy_by_segment_save_name,'wb') as f:
                    pickle.dump(attn_entropy_by_segment,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--segmentation_dir",default = '/work/09192/jianing/ls6/Memory_generation/behavior_data/segmentation')
    parser.add_argument("--save_dir", default = "/work/09192/jianing/ls6/Memory_generation/generated/")
    parser.add_argument("--file_name",default = 'recall_original_concat.pkl',choices = ['recall_original_concat.pkl','original_recall_concat.pkl','recall_tokens.pkl'])
    parser.add_argument("--story",default = 'pieman',help = 'to get the entropy of concatenated original stories (ie control), enter original')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--original_transcript_dir",default='/work/09192/jianing/ls6/Memory_generation/transcripts/moth_stories')
    parser.add_argument("--moth_output_dir",default = '/work/09192/jianing/ls6/Memory_generation/generated/')
    parser.add_argument("--simulation",action = 'store_true')
    parser.add_argument("--attention",action = 'store_true',help = 'compute total attention and attention entropy')
    parser.add_argument("--attention_by_event",action = 'store_true',help = 'compute attention by event')
    parser.add_argument("--entropy",action = 'store_true',help = 'compute entropy of tokens')
    parser.add_argument("--include_bos",action = 'store_true',help = 'whether to include bos token in the attention calculations')
    parser.add_argument("--model_recall",action = 'store_true')
    parser.add_argument("--attention_scale",type = float,help = 'controls entropy of model, larger value, more uniform attention')
    parser.add_argument("--temp",type = float,help = 'temperature to set for model generation')
    parser.add_argument("--att_to_story_start",action ='store_true',help = 'limit the modified attention to the start of story, not the start of sys prompt')
    parser.add_argument("--prompt_number",type = int,default = 0,help = 'prompt number')
    parser.add_argument("--verbatim",action='store_true',help = 'verbatim recall experiment')
    parser.add_argument("--model_recall_with_entropy",action = 'store_true',help = 'new model recalls with entropy computed')
    args = parser.parse_args()
    main(args)
