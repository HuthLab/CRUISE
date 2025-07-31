from transformers import AutoTokenizer, LlamaForCausalLM,LlamaConfig
import torch
import os
import pandas as pd
import numpy as np
import pickle
import glob
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils import model_to_path_dict

def form_df(output_by_scale):
    # Initialize lists to hold the data
    scales = []
    corrected_transcripts = []
    subjects = []

    # Iterate over the dictionary to populate the lists
    for scale, entries in output_by_scale.items():
        for entry in entries:
            scales.append(scale)
            corrected_transcripts.append(entry)

    # Create the subjects column
    subjects = list(range(1, len(corrected_transcripts) + 1))

    # Create the DataFrame
    df = pd.DataFrame({
        "scale": scales,
        "corrected transcript": corrected_transcripts,
        "subject": subjects
    })
    return df
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

def compute_att_entropy(layer_att,recall_start_index,include_bos = False,story_start_index = None):
    '''compute attention entropy for a single layer'''
    layer_att = layer_att.cpu()
    assert torch.allclose(torch.sum(layer_att,dim = -1),torch.ones_like(torch.sum(layer_att,dim = -1)),atol = 1e-3) # assuming the last dimension is the TO, it should add up to 1
    # if original-recall, want recall to original attention (ie after start idx to before start idx)
    # if recall-original, want original to recall attention (ie after start idx to before start idx)
    if story_start_index is not None:
        if layer_att.shape[2] >1:
            from_to_target_attn = layer_att[0,:,recall_start_index:,story_start_index:recall_start_index] # from recall start index, to the story
        else:
            from_to_target_attn = layer_att[0,:,:,story_start_index:recall_start_index] # num_heads,from chunk length, to chunk length
    else:
        if not include_bos: 
            from_to_target_attn = layer_att[0,:,:,1:recall_start_index] # num_heads,from chunk length, to chunk length
        else:
            from_to_target_attn = layer_att[0,:,:,:recall_start_index] # num_heads,from chunk length, to chunk length
    from_to_target_attn = from_to_target_attn.numpy()
    normalized_attn = normalize_attn(from_to_target_attn) # normalize attn for entropy calculation
    normalized_attn = np.nan_to_num(normalized_attn)
    layer_sum_attention = np.sum(from_to_target_attn,axis = ((-1,-2))) # nheads
    assert layer_sum_attention.shape[0] == from_to_target_attn.shape[0]
    log_att = np.log2(normalized_attn) # nheads,ntoken
    log_att[log_att == np.NINF] = 0 # sometimes attn can be 0, so log is inf. Need to get rid of them otherwise results are nan
    product = np.multiply(normalized_attn,log_att) # num_heads,from chunk length, to chunk length
    layer_entropy = -np.sum(product,axis = (-1,-2)) # sum over the tokens dimension, DIM should now be num_heads
    return layer_sum_attention,layer_entropy

def compute_entropy(scores):
    '''entropy of the generated tokens'''
    entropies = []
    for score in scores:
        # Convert scores to probabilities
        probs = torch.softmax(score, dim=-1)
        nonzero_probs = probs[torch.where(probs>0)] # otherwise entropy calculation yields nan
        # Compute entropy
        entropy = -(nonzero_probs * nonzero_probs.log()).sum(dim=-1)  # Sum over vocabulary
        entropies.append(entropy.item())  # Save entropy as a scalar
    return np.array(entropies)

def main(args):
    device = 'cuda'
    story = args.story
    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    model = LlamaForCausalLM.from_pretrained(model_to_path_dict[args.model]['hf_name'],attn_implementation="eager",device_map='auto',torch_dtype = torch.float16)
    print('model temp',model.generation_config.temperature)
    print('set temp to',args.temp)
    model.generation_config.temperature = args.temp
    print('new model temp',model.generation_config.temperature)
    save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'model_recall')
    with open(os.path.join(args.original_transcript_dir,'%s.txt'%story),'r') as f:
        original_txt = f.read()
    system_prompt_list = ["You are a human with typical memory ability, which means that you might not remember everything. You might only remember the gist of parts of the story, but not all of its details. You're going to listen to a story, and your task is to verbally recall the story in your own words in a verbal recording. When you describe the story, tell me everything you remember from the story (in as much detail as you can). Describe it in order from the beginning to the end. But if you later go back to an element that you forgot, that's OK. Do not rehearse your recall beforehand. Respond as if you’re speaking out loud.",
                          "You are a human with limited memory ability. You're going to listen to a story, and your task is to recall the story and summarize it in your own words in a verbal recording. Respond as if you’re speaking out loud.",
                          "You are a human with limited memory ability. You're going to listen to a story, and your task is to recall the story in your own words in as much detail as you can in a verbal recording. Respond as if you’re speaking out loud.",
                          ]
    system_prompt = system_prompt_list[args.prompt_number]
    
    user_prompt = "Here's the story: %s\nHere's your recall: "%original_txt
    if 'it' not in args.model and 'inst' not in args.model and 'chat' not in args.model:
        full_prompt = system_prompt + '\n'+user_prompt
        tokenized_chat = tokenizer(full_prompt, return_tensors="pt").input_ids
    else:
        print('instruct model')
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)
    recall_start_index = tokenized_chat.shape[1]-5
    if args.att_to_story_start:
        story_start_idx = -1
        for i,t in enumerate(tokenized_chat[0]):
            txt = tokenizer.decode(t)
            if ':' in txt:
                story_start_idx = i+1
                break
        assert story_start_idx>0 and story_start_idx<tokenized_chat.shape[1],'failed to find story start idx'
    else:
        story_start_idx = 1
    scales = [0,0.00001,0.00005,0.00007,0.0001,0.0002,0.0003,0.0004,0.0005,0.001,0.01] #[0.0002,0.0003,0.0004] #[0,0.00001,0.0001,0.0005,0.001,0.01] 
    output_by_scale = {}
    attention_by_scale = {}
    entropy_by_scale = {}
    for scale in scales:
        outputs = []
        mean_att_entropy = []
        entropies = []
        for i in tqdm(range(args.n)):
            if scale != 0:
                output = model.generate(tokenized_chat,attention_scale = scale,recall_start_index = recall_start_index,
                                        story_start_index=story_start_idx,max_new_tokens = 800,output_attentions=args.attention,
                                        return_dict_in_generate=True,output_scores=args.entropy)
            else:
                output = model.generate(tokenized_chat,story_start_index=story_start_idx,max_new_tokens = 800,
                                        output_attentions=args.attention,return_dict_in_generate=True,output_scores=args.entropy)
            sequence = output['sequences']
            outputs.append(tokenizer.decode(sequence[0][tokenized_chat.shape[1]:]))

            if args.entropy: # calculate entropy of each generated token
                scores = output['scores']
                assert sequence[0][tokenized_chat.shape[1]:].shape[0] == len(scores)
                entropy = compute_entropy(scores)
                entropies.append(entropy)

            if args.attention: # compute attention entropy
                all_token_att = [] # num generated tokens * layer * head
                for token_att in tqdm(output['attentions']):
                    all_layers_entropy = [] # layer * head
                    for layer_att in token_att:
                        _,layer_entropy = compute_att_entropy(layer_att,recall_start_index = recall_start_index,story_start_index = story_start_idx)
                        all_layers_entropy.append(layer_entropy)
                    all_token_att.append(np.array(all_layers_entropy))
                mean_all_token_att_entropy = np.mean(all_token_att,axis = 0) # layer * head
                mean_att_entropy.append(mean_all_token_att_entropy)

        output_by_scale[scale] = outputs
        attention_by_scale[scale] = mean_att_entropy
        entropy_by_scale[scale] = entropies
    df = form_df(output_by_scale)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.entropy:
        save_name = os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s_new.csv'%(story,model.generation_config.temperature,args.prompt_number,args.att_to_story_start))
        with open(os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s_new_output_entropy.pkl'%(story,model.generation_config.temperature,args.prompt_number,args.att_to_story_start)),'wb')as f:
            pickle.dump(entropy_by_scale,f)
    else:
        save_name = os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s.csv'%(story,model.generation_config.temperature,args.prompt_number,args.att_to_story_start))
    if os.path.exists(save_name):
        old_df = pd.read_csv(save_name)
        new_df = pd.concat([old_df, df])
        new_df['subject'] = np.arange(1,len(new_df)+1)
        df= new_df
    df.to_csv(save_name,index = False)
    if args.attention:
        with open(os.path.join(save_dir,'%s_model_recall_transcript_temp%.2f_prompt%d_att_to_story_start_%s_att_entropy.pkl'%(story,model.generation_config.temperature,args.prompt_number,args.att_to_story_start)),'wb')as f:
            pickle.dump(attention_by_scale,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir", default = "/work/09192/jianing/ls6/Memory_generation/generated/")
    parser.add_argument("--original_transcript_dir",default='/work/09192/jianing/ls6/Memory_generation/transcripts/moth_stories')
    parser.add_argument("--moth_output_dir",default = '/work/09192/jianing/ls6/Memory_generation/generated/')
    parser.add_argument("--story",default = 'pieman',help = 'to run the concatenated entropy of original stories, enter original')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--n",type = int,help = 'number of samples to generate',default = 50)
    parser.add_argument("--temp",type = float,help = 'temperature to set for model generation')
    parser.add_argument("--att_to_story_start",action ='store_true',help = 'limit the modified attention to the start of story, not the start of sys prompt')
    parser.add_argument("--prompt_number",type = int,default = 0,help = 'prompt number')
    parser.add_argument("--attention",action = 'store_true',help = 'output attentions at generation and calculate attention entropy')
    parser.add_argument("--entropy",action = 'store_true',help = 'calculate entropy (NOT CE) of the generated tokens')
    args = parser.parse_args()
    main(args)