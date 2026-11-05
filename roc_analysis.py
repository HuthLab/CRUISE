from sklearn.metrics import auc,precision_recall_curve,roc_auc_score
from scipy.signal import correlate,correlation_lags
from statsmodels.tsa.stattools import acf
from transformers import AutoTokenizer
import torch 
from scipy import stats
import numpy as np
import pickle
import os
import re
import glob
import string
from torch.nn import functional as F
from natsort import os_sorted
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from utils import model_to_path_dict

def plot_all_subjects_roc(roc_dir,roc_files,fig_dir):
    # fig,ax = plt.subplots(figsize = (7,7))
    all_auc = {}
    for roc_file in roc_files:
        subject = int(re.findall(r'\d+',roc_file)[0])
        with open(os.path.join(roc_dir,roc_file),'rb') as f:
            sub_roc_dict = pickle.load(f)
        fpr = sub_roc_dict['fpr']
        tpr = sub_roc_dict['tpr']
        # ax.plot(fpr,tpr,alpha = 0.5)
        all_auc[subject] = auc(fpr,tpr)

    # xx = np.linspace(0,1,100)
    # ax.plot(xx,xx)
    # ax.set_xlabel('false positive rate')
    # ax.set_ylabel('true positive rate')
    # ax.set_title('ROC')
    # fig.savefig(os.path.join(fig_dir,'all_subjects_roc.png'))
    return all_auc

def auc_hist(all_auc,fig_dir,pr = False):
    all_auc_list = list(all_auc.values())
    # plt.figure()
    # plt.hist(all_auc_list)
    # plt.ylabel('Frequency')
    if pr:
        print('Mean precision recall AUC',np.mean(all_auc_list))
        print('Median precision recall AUC',np.median(all_auc_list))
        # plt.title('Precision-recall AUC')
        # plt.xlabel('Precision-recall AUC')
        # plt.savefig(os.path.join(fig_dir,'pr_auc_histogram.png'))
    else:
        print('Mean AUC',np.mean(all_auc_list))
        print('Median AUC',np.median(all_auc_list))
        # plt.title('AUC')
        # plt.xlabel('AUC')
        # plt.savefig(os.path.join(fig_dir,'auc_histogram.png'))


def plot_precision_recall(roc_dir,roc_files,fig_dir):
    all_pr_auc = {}
    # fig,ax = plt.subplots(figsize = (7,7))
    for roc_file in roc_files:
        subject = int(re.findall(r'\d+',roc_file)[0])
        with open(os.path.join(roc_dir,roc_file),'rb') as f:
            sub_roc_dict = pickle.load(f)
        precision = sub_roc_dict['precision']
        recall = sub_roc_dict['recall']
        # ax.plot(recall,precision,alpha = 0.5)
        sub_auc = auc(recall, precision)
        all_pr_auc[subject]=sub_auc

    # ax.set_xlabel('recall')
    # ax.set_ylabel('precision')
    # ax.set_title('Precision-recall curve')
    # fig.savefig(os.path.join(fig_dir,'all_subjects_pr.png'))
    return all_pr_auc

def auc_score_block_permutation_instruct(newline_prob,input_tokenized,newline_id,story_start_index_token,n_perm,block_size,pr_auc = False,instruct = True,inst_ending_length = None):
    '''
    perform block permutation of the models' prediction for a subject's segmentation
    pr_auc: if true, calculate AUC for precision vs. recall. If false, calculate the original AUC for FP vs. TP
    returns list of permuted auc values of length n_perm
    '''
    # only consider probabilities after the start of the story 
    human_segmentation_bool = torch.squeeze(input_tokenized==newline_id).detach().cpu() # input len 
    if instruct:
        human_segmentation_bool = human_segmentation_bool[story_start_index_token:-inst_ending_length].numpy() # story len
        story_newline_prob = newline_prob[story_start_index_token-1:-inst_ending_length-1].detach().cpu().numpy() # story len * 1
    else:
        human_segmentation_bool = human_segmentation_bool[story_start_index_token:].numpy() # story len
        story_newline_prob = newline_prob[story_start_index_token-1:-1].detach().cpu().numpy() # story len * 1
    assert human_segmentation_bool.shape == story_newline_prob.shape, 'shape of human segmentation and model segmentation must be the same'
    
    block_start_indices = np.arange(0,len(human_segmentation_bool),block_size).astype(int)
    
    # the last block might be smaller than the pre-determined block size 
    actual_block_size = np.repeat(block_size,len(block_start_indices))
    actual_block_size[-1] = len(human_segmentation_bool)-block_start_indices[-1]
    
    permuted_auc = np.zeros(n_perm)-1
    rng = np.random.default_rng(42)
    for i in range(n_perm):
        permuted_human_segmentation_bool = np.ones_like(human_segmentation_bool)*(-99)
        permuted_indices = rng.permutation(len(block_start_indices))
        permuted_block_start_indices = block_start_indices[permuted_indices]
        permuted_actual_block_size = actual_block_size[permuted_indices]
        assert permuted_block_start_indices.shape == block_start_indices.shape
        assert permuted_actual_block_size.shape == actual_block_size.shape
        #print(len(permuted_block_start_indices))
        filled_ind = 0 # start index of this block
        for chunk_ind,permuted_start_ind in enumerate(permuted_block_start_indices):
            this_block_size = permuted_actual_block_size[chunk_ind] # size of the chunk after permutation
            permuted_human_segmentation_bool[filled_ind:filled_ind+this_block_size]=human_segmentation_bool[permuted_start_ind:permuted_start_ind+this_block_size]
            filled_ind = filled_ind+permuted_actual_block_size[chunk_ind] # update for next block 
        assert sum(permuted_human_segmentation_bool==-99)==0,'all indices in permuted_human_segmentation_bool must be filled'
        
        if pr_auc: # auc of precision vs. recall 
            precision, recall, pr_thresholds = precision_recall_curve(y_true = permuted_human_segmentation_bool,y_score=story_newline_prob)
            this_auc = auc(recall, precision)
        else: # original auc 
            this_auc = roc_auc_score(y_true = permuted_human_segmentation_bool,y_score = story_newline_prob)
        permuted_auc[i]=this_auc
    assert sum((permuted_auc)==-1)==0,'no entries of permuted_auc should be -1'
    return permuted_auc

def permute_auc(logit_dir,all_auc,all_pr_auc,segmentation_df,tokenizer,fig_dir,newline_id,instruct,threshold,inst_ending_length = None):
    '''
    AUC permutation for all subjects
    Saves p-value histograms
    returns: 
        all_permuted_auc: dict of list, n_perm permuted AUC values for each subject 
        pvals: dict of float, P(permuted AUC > model's AUC)
    '''
    prompt = '''An event is an ongoing coherent situation. In the following story, the end of an event and the beginning of the next is indicated by a new line. This is the story:''' 
    n_perm = 1000
    all_permuted_auc = {}
    all_permuted_pr_auc = {}
    pvals = {}
    pvals_pr = {}
    block_sizes = []
    # calculate median block size 
    for subject in segmentation_df.index:
        block_size = 1
        subject_id = segmentation_df['subject'][subject]
        # logits = torch.load(os.path.join(logit_dir,'logits_sub%d.pkl'%subject_id))
        # prob = F.softmax(logits,dim = -1)
        #newline_prob = prob[0,:,newline_id].detach().cpu() # shape: input token len
        newline_prob = torch.load(os.path.join(logit_dir,'newline_prob_sub%d.pkl'%subject_id))
        # auto_corr = correlate(newline_prob/np.linalg.norm(newline_prob),newline_prob/np.linalg.norm(newline_prob))
        # while auto_corr[int(len(auto_corr)/2)+block_size]>threshold:
        #     block_size +=1
        auto_corr_new = acf(newline_prob, nlags=400) # 400 should be enough to go below the threshold
        while auto_corr_new[block_size]>threshold:
            block_size+=1
        block_sizes.append(block_size)
    median_block_size = int(np.median(block_sizes))
    print('median block size: %d'%median_block_size)
    for subject in tqdm(segmentation_df.index):
        subject_id = int(segmentation_df['subject'][subject])
        subject_segmentation = segmentation_df['segmented_txt_no_space'][subject]
        actual_prompt = prompt+subject_segmentation
        if instruct:
            prompt = '''An event is an ongoing coherent situation. In the following story, the end of an event and the beginning of the next is indicated by a new line. This is the story: ''' 
            actual_prompt = prompt+subject_segmentation
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.",
                },
                {"role": "user", "content": actual_prompt},
            ]
            # chat template already contains bos token 
            prompt_w_template = tokenizer.apply_chat_template(messages, tokenize=False) 
            tokenized_txt = tokenizer.tokenize(prompt_w_template)
            input_tokenized = tokenizer.apply_chat_template(messages, tokenize=True,return_tensors="pt")
            input_tokenized2 = tokenizer(prompt_w_template, return_tensors="pt",add_special_tokens=False).input_ids # don't add bos token, it's already added in the template
            # deal with new version of transformers: in new versions the token ids from apply_chat_template has a separate field, it doesn't just return the tensor
            input_tokenized = input_tokenized.input_ids if hasattr(input_tokenized, "input_ids") else input_tokenized
            assert torch.allclose(input_tokenized,input_tokenized2), 'two methods of tokenization should be the same'

            story_start_index_txt = tokenized_txt.index(":")+1 # both llama2 and llama3 tokenize : as a separate token 
            story_start_index_token = story_start_index_txt # no more +1 here because the tokenized_txt already has bos

        else:
            prompt_tokenized = tokenizer(prompt, return_tensors="pt").input_ids
            subject_tokenized = tokenizer(subject_segmentation, return_tensors="pt",add_special_tokens = False).input_ids
            input_tokenized = torch.cat((prompt_tokenized,subject_tokenized),axis = 1)
            story_start_index_token = prompt_tokenized.shape[1]
        # logits = torch.load(os.path.join(logit_dir,'logits_sub%d.pkl'%subject_id))
        # prob = F.softmax(logits,dim = -1)
        # newline_prob = prob[0,:,newline_id].detach().cpu() # shape: input token len 
        newline_prob = torch.load(os.path.join(logit_dir,'newline_prob_sub%d.pkl'%subject_id))
        # regular AUC
        if instruct:
            sub_permuted_auc = auc_score_block_permutation_instruct(newline_prob,input_tokenized,newline_id,story_start_index_token,n_perm,median_block_size,instruct = True,inst_ending_length = inst_ending_length)
        else:
            sub_permuted_auc = auc_score_block_permutation_instruct(newline_prob,input_tokenized,newline_id,story_start_index_token,n_perm,median_block_size,instruct = False)
        all_permuted_auc[subject_id] = sub_permuted_auc
        # calculate pvalue
        sub_auc = all_auc[subject_id]
        pval = sum(sub_permuted_auc>sub_auc)/len(sub_permuted_auc)
        pvals[subject_id] = pval
        
        # precision_recall auc 
        if instruct:
            sub_permuted_pr_auc = auc_score_block_permutation_instruct(newline_prob,input_tokenized,newline_id,story_start_index_token,n_perm,median_block_size,pr_auc = True,instruct = True,inst_ending_length = inst_ending_length)
        else:
            sub_permuted_pr_auc = auc_score_block_permutation_instruct(newline_prob,input_tokenized,newline_id,story_start_index_token,n_perm,median_block_size,pr_auc = True,instruct = False)
        all_permuted_pr_auc[subject_id] = sub_permuted_pr_auc
        sub_pr_auc = all_pr_auc[subject_id]
        pval_pr = sum(sub_permuted_pr_auc>sub_pr_auc)/len(sub_permuted_pr_auc)
        pvals_pr[subject_id] = pval_pr

    pvals_arr = np.array(list(pvals.values()))
    print('mean:%.2f,median:%.2f,standard error:%.2f'%(np.mean(pvals_arr),np.median(pvals_arr),stats.sem(pvals_arr)))
    print('Proportion with pval <= 0.05:%.3f'%(sum(pvals_arr<0.05)/len(pvals_arr)))

    pvals_pr_arr = np.array(list(pvals_pr.values()))
    print('mean:%.2f,median:%.2f,standard error:%.2f'%(np.mean(pvals_pr_arr),np.median(pvals_arr),stats.sem(pvals_pr_arr)))
    print('Proportion with precision-recall pval <= 0.05:%.3f'%(sum(pvals_pr_arr<0.05)/len(pvals_pr_arr)))
    
    # plt.figure()
    # plt.hist(pvals_arr)
    # plt.title('histogram of p values for AUC')
    # plt.xlabel('p value')
    # plt.ylabel('Subject frequency')
    # plt.savefig(os.path.join(fig_dir,'auc_pval_hist.png'))

    # plt.figure()
    # plt.hist(pvals_pr_arr)
    # plt.title('histogram of p values for precision-recall AUC')
    # plt.xlabel('p value')
    # plt.ylabel('Subject frequency')
    # plt.savefig(os.path.join(fig_dir,'auc_pr_pval_hist.png'))
    return all_permuted_auc,all_permuted_pr_auc,pvals,pvals_pr,median_block_size



def main(args):
    save_dir = os.path.join(args.save_dir,model_to_path_dict[args.model]['save_dir_name'],'prolific_data')
    segmentation_df = pd.read_csv(os.path.join(args.segmentation_dir,args.story, args.story+'_segmentation.csv'))
    if args.exclusion:
        story_exclusion = pd.read_csv(os.path.join(args.exclusion_dir,'%s_exclusion.csv'%args.story))
        excluded_subjects = story_exclusion['excluded'].loc[story_exclusion['segmentation']==1].values
        segmentation_df = segmentation_df[~segmentation_df['subject'].isin(excluded_subjects)]

    if args.instruct:
        logit_dir = os.path.join(save_dir,args.story+'_instruct','subject_segmentation_logits')
        roc_dir = os.path.join(save_dir,args.story+'_instruct','roc')
        fig_dir = os.path.join(save_dir,args.story+'_instruct','figure')
    else:
        logit_dir = os.path.join(save_dir,args.story,'subject_segmentation_logits')
        roc_dir = os.path.join(save_dir,args.story,'roc')
        fig_dir = os.path.join(save_dir,args.story,'figure')
    # if not os.path.exists(fig_dir):
        # os.makedirs(fig_dir)
    roc_files = os.listdir(roc_dir)
    roc_files = os_sorted(roc_files)
    if args.exclusion:
        roc_files = [f for f in roc_files if int(re.findall(r'\d+',f)[0]) not in excluded_subjects]
    # for roc_file in roc_files:
    #     subject = int(re.findall(r'\d+',roc_file)[0])
    # calculate AUC
    print('calculating AUC')
    all_auc = plot_all_subjects_roc(roc_dir,roc_files,fig_dir)
    all_pr_auc = plot_precision_recall(roc_dir,roc_files,fig_dir)
    # plot AUC histogram and save
    auc_hist(all_auc,fig_dir)
    auc_hist(all_pr_auc,fig_dir,pr=True)

    tokenizer = AutoTokenizer.from_pretrained(model_to_path_dict[args.model]['hf_name'])
    if args.model == 'Llama-2-13b-chat-hf':
        newline_id = tokenizer('\n', add_special_tokens=False).input_ids[1]
        inst_ending_length = 4
    elif args.model == 'gemma-7b':
        newline_id = tokenizer('\n', add_special_tokens=False).input_ids[0]
        inst_ending_length = None
    elif args.model=='Llama3-8b-instruct':
        newline_id = tokenizer('\n', add_special_tokens=False).input_ids[0]
        inst_ending_length = 1
    
    # permutation
    print('permutation')
    
    if args.instruct:
        all_permuted_auc,all_permuted_pr_auc,pvals,pvals_pr,median_block_size=permute_auc(logit_dir,all_auc,all_pr_auc,segmentation_df,tokenizer,fig_dir,newline_id,instruct = True,threshold = args.threshold,inst_ending_length = inst_ending_length)
        perm_save_dir = os.path.join(save_dir,args.story+'_instruct','permutation_results')
    else:
        all_permuted_auc,all_permuted_pr_auc,pvals,pvals_pr,median_block_size=permute_auc(logit_dir,all_auc,all_pr_auc,segmentation_df,tokenizer,fig_dir,newline_id,instruct = False,threshold = args.threshold)
        perm_save_dir= os.path.join(save_dir,args.story,'permutation_results')
    if not os.path.exists(perm_save_dir):
        os.makedirs(perm_save_dir)
    with open(os.path.join(perm_save_dir,'all_permuted_auc.pkl'),'wb') as f:
        pickle.dump(all_permuted_auc,f)
    with open(os.path.join(perm_save_dir,'all_permuted_pr_auc.pkl'),'wb') as f:
        pickle.dump(all_permuted_pr_auc,f)
    with open(os.path.join(perm_save_dir,'permutation_pvals.pkl'),'wb') as f:
        pickle.dump(pvals,f)
    with open(os.path.join(perm_save_dir,'permutation_pr_pvals.pkl'),'wb') as f:
        pickle.dump(pvals_pr,f)
    
    with open(os.path.join(perm_save_dir,'all_auc.pkl'),'wb') as f:
        pickle.dump(all_auc,f)
    with open(os.path.join(perm_save_dir,'all_pr_auc.pkl'),'wb') as f:
        pickle.dump(all_pr_auc,f)
    auc_print = "mean AUC: %.2f"%np.mean(list(all_auc.values()))
    pvals_arr = np.array(list(pvals.values()))
    pval_print = 'pval mean:%.2f,median:%.2f,standard error:%.2f,\nProportion with pval <= 0.05:%.3f'%(np.mean(pvals_arr),np.median(pvals_arr),stats.sem(pvals_arr),sum(pvals_arr<0.05)/len(pvals_arr))
    pr_print = "mean p-r AUC: %.2f"%np.mean(list(all_pr_auc.values()))
    pvals_pr_arr = np.array(list(pvals_pr.values()))
    pvals_pr_print = 'pval mean:%.2f,median:%.2f,standard error:%.2f,\nProportion with precision-recall pval <= 0.05:%.3f'%(np.mean(pvals_pr_arr),np.median(pvals_pr_arr),stats.sem(pvals_pr_arr),sum(pvals_pr_arr<0.05)/len(pvals_pr_arr))
    
    with open(os.path.join(perm_save_dir,'auc_results.txt'), 'w') as file:
        file.write(auc_print + '\n')  
        file.write('permutation auto corr threshold: %.2f'%args.threshold+ '\n')
        file.write('permutation block size: %d'%median_block_size+ '\n')
        file.write(pval_print + '\n') 
    with open(os.path.join(perm_save_dir,'pr_auc_results.txt'), 'w') as file:
        file.write(pr_print + '\n')  
        file.write('permutation auto corr threshold: %.2f'%args.threshold+ '\n')
        file.write('permutation block size: %d'%median_block_size + '\n')
        file.write(pvals_pr_print + '\n')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir", default = "../generated/")
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--story",default = 'pieman')
    parser.add_argument("--model",choices = ['Llama-2-13b-chat-hf','gemma-7b','Llama3-8b-instruct'],default = 'Llama3-8b-instruct')
    parser.add_argument('--instruct',action ='store_true')
    parser.add_argument('--threshold',type = float, default = 0.1)
    parser.add_argument('--exclusion',action='store_true')
    parser.add_argument('--exclusion_dir',default='../behavior_data/exclusion')
    args = parser.parse_args()
    main(args)
