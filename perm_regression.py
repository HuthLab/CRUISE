from transformers import AutoTokenizer
import torch 
import numpy as np
import pandas as pd
import pickle
import os
import re
import glob
import string
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from tqdm.notebook import tqdm
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score,roc_curve,r2_score
from statsmodels.tsa.stattools import acf
from utils import get_segmentation_indices,segmentation_to_word_list,model_to_path_dict
from sklearn.linear_model import  Lasso,Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit,GroupKFold,LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from itertools import combinations, product
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import argparse 

def find_word_indices_in_tokenized_txt(original_txt,tokenized_txt,model_initial_char):
    '''
    input: original_txt (string, story transcript), tokenized_txt (list, tokenized by LM)
    output: list of lists, each sublist indicates the [start,end] (exclusive) indices or a word from the original_txt in the tokenized_txt'''
    start_idx_in_tokenized_txt = 0
    end_idx_in_tokenized_txt = 0
    curr_word = ''
    word_indices_in_tokenized_txt = []
    for w in original_txt.split(' '):
        while curr_word!=w and len(curr_word)<len(w):
            curr_word+=tokenized_txt[end_idx_in_tokenized_txt].translate(str.maketrans('', '', model_initial_char))
            end_idx_in_tokenized_txt+=1
        if curr_word==w:
            if curr_word =='':
                curr_word+=tokenized_txt[end_idx_in_tokenized_txt].translate(str.maketrans('', '', model_initial_char))
                assert curr_word==w
                end_idx_in_tokenized_txt+=1
            word_indices_in_tokenized_txt.append([start_idx_in_tokenized_txt,end_idx_in_tokenized_txt])
            start_idx_in_tokenized_txt=end_idx_in_tokenized_txt
            curr_word = ''
        else:
            print('%s not found'%w)
    return word_indices_in_tokenized_txt

def build_signal_mat2(signal,nfeat):
    '''adapted from Kumar et al. to go to time point t'''
    feat_win=nfeat
    mat_indx=np.zeros([len(signal),feat_win]).astype(int)
    sig_mat=np.empty([len(signal),feat_win])
    for i in range(feat_win,len(signal)):
        mat_indx[i]=np.linspace(i-feat_win+1,i,feat_win)
        
        sig_mat[i]=np.squeeze(signal[mat_indx[i,:]])
    return sig_mat
def z_score(signal):
    scaler=StandardScaler()
    signal=signal.reshape(-1,1)
    scaler.fit(signal)
    signal_z=scaler.transform(signal)
    return signal_z


def _permute_story_signals_and_regress(perm, stories, all_stories_sig, story_id_all, button_all, args):
    perm_sig_all = []
    for story in stories:
        story_sig_arr = all_stories_sig[story]
        perm_sig = np.roll(story_sig_arr, perm)
        perm_sig = build_signal_mat2(perm_sig, args.nfeat)
        perm_sig = perm_sig[args.nfeat:, :]
        perm_sig_all.append(perm_sig)
    perm_sig_all = np.vstack(perm_sig_all)
    assert button_all.shape[0] == perm_sig_all.shape[0]
    corr_perm, betas_perm, result, cvres, best_models = compute_grid_search_regression(
        perm_sig_all, story_id_all, button_all, args
    )
    return corr_perm, betas_perm


def _run_permutation_regressions(lin_s, stories, all_stories_sig, story_id_all, button_all, args, use_parallel=False, n_jobs=1):
    if not use_parallel:
        corr_perm_all = []
        betas_perm_all = []
        for perm in tqdm(lin_s, total=len(lin_s)):
            corr_perm, betas_perm = _permute_story_signals_and_regress(
                perm, stories, all_stories_sig, story_id_all, button_all, args
            )
            corr_perm_all.append(corr_perm)
            betas_perm_all.append(betas_perm)
        return corr_perm_all, betas_perm_all

    perm_results = Parallel(
        n_jobs=n_jobs,
        prefer='threads',
        verbose=0,
    )(
        delayed(_permute_story_signals_and_regress)(
            perm, stories, all_stories_sig, story_id_all, button_all, args
        )
        for perm in tqdm(lin_s, total=len(lin_s))
    )
    corr_perm_all = [result[0] for result in perm_results]
    betas_perm_all = [result[1] for result in perm_results]
    return corr_perm_all, betas_perm_all


def auc_score_block_permutation_instruct(pred,human_segmentation_bool,n_perm,block_size,pr_auc = False):
    '''
    perform block permutation of the models' prediction for a subject's segmentation
    pr_auc: if true, calculate AUC for precision vs. recall. If false, calculate the original AUC for FP vs. TP
    returns list of permuted auc values of length n_perm
    '''
    # only consider probabilities after the start of the story     
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
            precision, recall, pr_thresholds = precision_recall_curve(y_true = permuted_human_segmentation_bool,y_score=pred)
            this_auc = auc(recall, precision)
        else: # original auc 
            this_auc = roc_auc_score(y_true = permuted_human_segmentation_bool,y_score = pred)
        permuted_auc[i]=this_auc
    assert sum((permuted_auc)==-1)==0,'no entries of permuted_auc should be -1'
    return permuted_auc


def compute_grid_search_regression(sig_mat_all,story_id_all, button_all, args):
    '''adapted from Kumar et al.'''
    X, y=sig_mat_all, button_all
    ps = PredefinedSplit(story_id_all)
    corr_score=[]
    betas=[]
    cvres = []
    #p_space1 = [10000, 20000, 30000, 40000]#np.logspace(0.01, 20, num=10, endpoint=True)
    #p_space2 = np.logspace(0.00001, 4, num=20, endpoint=True)
    p_space3 = np.logspace(-16, -3, num=20, endpoint=True)
    p_space4 = np.logspace(-7,0, num=20, endpoint=True)
    p_space5 = np.logspace(-5,5, num=20, endpoint=True)
    if args.reg_model=='Lasso':
        parameters=[ {
                'clf': [Lasso()],
                'alpha': p_space3,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-2]

            }]
    if args.reg_model=='Ridge':
        parameters=[ {
                'clf': [Ridge()],
                'alpha': p_space4,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-2]

            }]
    if args.reg_model=='Logistic':
        parameters=[ {
                'clf': [LogisticRegression()],
                'penalty': ['l2'],
                'C':p_space5,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-4]

            }]
    result=[]
    y_preds = []
    best_models = []
    for params in parameters:

        #classifier
        clf = params['clf'][0]
            #getting arguments by
            #popping out classifier
        params.pop('clf')
       
        for train_index, test_index in ps.split():
            # split the data 
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
#             print(X_train.shape,X_test.shape)
#             print('train',train_index,'test',test_index)
            story_train, story_test=story_id_all[train_index],story_id_all[test_index]
            ps_train=PredefinedSplit(story_train)
        
            story_train = story_id_all[train_index]
            inner_stories = np.unique(story_train)
            # --- Create inner CV: all 4-train / 3-val combinations ---
            inner_cv = []
            for val_stories in combinations(inner_stories, 3):
                val_idx = np.where(np.isin(story_train, val_stories))[0]
                inner_train_idx = np.where(~np.isin(story_train, val_stories))[0]
                inner_cv.append((inner_train_idx, val_idx))
        
            # fit the model on the training set 
            #search = GridSearchCV(clf,param_grid=params, n_jobs=-1,cv = inner_cv,scoring = 'neg_mean_squared_error')
            if args.reg_model=='Logistic':
                search = GridSearchCV(clf,param_grid=params, n_jobs=-1, cv=ps_train)
            else:
                search = GridSearchCV(clf,param_grid=params, n_jobs=-1, cv=ps_train,scoring = 'neg_mean_squared_error')
#             print("Std of outer y_test:",np.std(y_test))
#             print("Std of y_train across folds:", [np.std(y[train]) for train, _ in ps_inner.split()])
#             print("Std of y_test across folds:", [np.std(y[test]) for _, test in ps_inner.split()])


            search.fit(X_train, y_train)
            fold_res = pd.DataFrame(search.cv_results_)
                
            # calculate the accuracy for the hold out run
            best_model=search.best_estimator_
            print(best_model)
            betas.append(best_model.coef_)
            ypred= best_model.predict(X_test)
            y_preds.append(ypred)
            if args.stat=='Pearson':
                corr_score.append(np.corrcoef(y_test,ypred)[0,1])
            elif args.stat=='r2':
                corr_score.append(best_model.score(X_test,y_test))
            elif args.stat =='auc':
                y_prob = best_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                corr_score.append(auc)
            result.append\
            (   {
            'grid': search,
            'classifier': search.best_estimator_,
            'best params': search.best_params_,
            'cv': search.cv,
            'corr':corr_score,
            'betas':best_model.coef_
                
            })
            cvres.append(fold_res)
            best_models.append(best_model)

    return corr_score, betas,result,cvres,best_models


def compute_grid_search_regression_individual_subject_test(sig_mat_all,story_id_all, button_all, args,story_filtered_subject_segmentation_bool,n_perm,threshold):
    '''
    adapted from Kumar et al. 
    During test, compute the AUC for each individual subject's segmentation, and compute p-value of the AUC by comparing to permuted AUC scores obtained from block permutation of the model's prediction.
    '''
    X, y=sig_mat_all, button_all
    ps = PredefinedSplit(story_id_all)
    corr_score=[]
    betas=[]
    cvres = []
    #p_space1 = [10000, 20000, 30000, 40000]#np.logspace(0.01, 20, num=10, endpoint=True)
    #p_space2 = np.logspace(0.00001, 4, num=20, endpoint=True)
    p_space3 = np.logspace(-16, -3, num=20, endpoint=True)
    p_space4 = np.logspace(-7,0, num=20, endpoint=True)
    p_space5 = np.logspace(-5,5, num=20, endpoint=True)
    if args.reg_model=='Lasso':
        parameters=[ {
                'clf': [Lasso()],
                'alpha': p_space3,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-2]

            }]
    if args.reg_model=='Ridge':
        parameters=[ {
                'clf': [Ridge()],
                'alpha': p_space4,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-2]

            }]
    if args.reg_model=='Logistic':
        parameters=[ {
                'clf': [LogisticRegression()],
                'penalty': ['l2'],
                'C':p_space5,
                'max_iter':[5000],
                #'tol':[1e-4]
                'tol':[1e-4]

            }]
    result=[]
    y_preds = []
    best_models = []
    test_story_auc = {}
    test_story_pval = {}
    test_story_tpr_fpr = {}
    for params in parameters:

        #classifier
        clf = params['clf'][0]
            #getting arguments by
            #popping out classifier
        params.pop('clf')
       
        for train_index, test_index in ps.split():
            # split the data 
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
#             print(X_train.shape,X_test.shape)
#             print('train',train_index,'test',test_index)
            story_train, story_test=story_id_all[train_index],story_id_all[test_index]
            ps_train=PredefinedSplit(story_train)
        
            story_train = story_id_all[train_index]
            inner_stories = np.unique(story_train)
            # --- Create inner CV: all 4-train / 3-val combinations ---
            inner_cv = []
            for val_stories in combinations(inner_stories, 3):
                val_idx = np.where(np.isin(story_train, val_stories))[0]
                inner_train_idx = np.where(~np.isin(story_train, val_stories))[0]
                inner_cv.append((inner_train_idx, val_idx))
        
            # fit the model on the training set 
            #search = GridSearchCV(clf,param_grid=params, n_jobs=-1,cv = inner_cv,scoring = 'neg_mean_squared_error')
            if args.reg_model=='Logistic':
                search = GridSearchCV(clf,param_grid=params, n_jobs=-1, cv=ps_train)
            else:
                search = GridSearchCV(clf,param_grid=params, n_jobs=-1, cv=ps_train,scoring = 'neg_mean_squared_error')
#             print("Std of outer y_test:",np.std(y_test))
#             print("Std of y_train across folds:", [np.std(y[train]) for train, _ in ps_inner.split()])
#             print("Std of y_test across folds:", [np.std(y[test]) for _, test in ps_inner.split()])


            search.fit(X_train, y_train)
            fold_res = pd.DataFrame(search.cv_results_)
                
            # calculate the accuracy for the hold out run
            best_model=search.best_estimator_
            print(best_model)
            betas.append(best_model.coef_)
            ypred= best_model.predict(X_test)
            y_preds.append(ypred)
            if args.stat=='Pearson':
                corr_score.append(np.corrcoef(y_test,ypred)[0,1])
            elif args.stat=='r2':
                corr_score.append(best_model.score(X_test,y_test))
            elif args.stat =='auc':
                y_prob = best_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                corr_score.append(auc)
                
            test_story_index = np.unique(story_id_all[test_index])
            assert len(test_story_index)==1
            test_story_index = test_story_index[0]
            test_story_subject_segmentation_bool = story_filtered_subject_segmentation_bool[test_story_index]
            assert np.all(test_index == sorted(test_index)) # makes sure the feature indices are sorted in order 
            this_story_auc = []
            this_story_pval = []
            this_story_tpr_fpr = []

            block_size = 2 # start with block size of 2, and determine block size by looking at the auto-correlation of the model's prediction. We want to choose a block size so that the auto-correlation between permuted segments is below a certain threshold (e.g. 0.1)
            auto_corr_new = acf(ypred, nlags=400) # 400 should be enough to go below the threshold
            while auto_corr_new[block_size]>threshold:
                block_size+=1
            for sub_bool in test_story_subject_segmentation_bool:
                assert len(ypred)==len(sub_bool)
                sub_auc = roc_auc_score(sub_bool,ypred)
                this_story_auc.append(sub_auc)
                sub_permuted_auc = auc_score_block_permutation_instruct(ypred,sub_bool,n_perm,block_size = block_size,pr_auc = False)
                pval = sum(sub_permuted_auc>sub_auc)/len(sub_permuted_auc)
                this_story_pval.append(pval)
                fpr, tpr, roc_thresholds = roc_curve(y_true = sub_bool,y_score = ypred)
                this_story_tpr_fpr.append({'fpr':fpr,'tpr':tpr})
            test_story_auc[test_story_index] = this_story_auc
            test_story_pval[test_story_index] = this_story_pval
            test_story_tpr_fpr[test_story_index] = this_story_tpr_fpr
            
            
            result.append\
            (   {
            'grid': search,
            'classifier': search.best_estimator_,
            'best params': search.best_params_,
            'cv': search.cv,
            'corr':corr_score,
            'betas':best_model.coef_
                
            })
            cvres.append(fold_res)
            best_models.append(best_model)

    return corr_score, betas,result,cvres,best_models,test_story_auc,test_story_pval,test_story_tpr_fpr

def get_human_segmentation_prop(stories,moth_output_dir,segmentation_dir,original_transcript_dir,model_initial_char,by_subject = True):
    story_word_indices_in_tokenized_txt = {}
    story_human_segmentation_prop = {}
    story_human_segmentation_prop_smoothed = {}
    story_subject_segmentation_bool = {}
    for story in stories:
        with open(os.path.join(moth_output_dir,story,'tokenized_txt.pkl'),'rb') as f:
            tokenized_txt = pickle.load(f)
        story_tokens = torch.load(os.path.join(moth_output_dir,story,'tokens.pkl'),map_location=torch.device('cpu'))
        story_tokens = story_tokens[0,1:].cpu().detach()
        consensus_path = os.path.join(segmentation_dir,story,'%s_consensus.txt'%story)
        with open(consensus_path,'r') as f:
            consensus_txt = f.read()
        consensus_txt = consensus_txt.split('\n')
        with open(os.path.join(original_transcript_dir,'%s.txt'%story),'r') as f:
            original_txt = f.read()
        consensus_wordlist = segmentation_to_word_list(consensus_txt)
        with open(os.path.join(segmentation_dir,story,'%s_segmentation_prop.pkl'%story),'rb') as f:
            human_segmentation_proportion = pickle.load(f)
        if len(human_segmentation_proportion) != len(original_txt.split(' ')): # for adventuresinsayingyes there's an extra space in the txt of the segmentation df. This doesn't affect exisiting analysis because we got rid of spaces when converting to token indices
            segmentation_data_exp = glob.glob(os.path.join(segmentation_dir,story,'%s_segmentation_data_exp*.csv'%story))
            segmentation_all_subs = pd.read_csv(segmentation_data_exp[0])
            original_txt_old = segmentation_all_subs['original_txt'][0]
            if original_txt_old[0]==' ' and original_txt[0]!= ' ':
                assert original_txt_old[1:] == original_txt,'old txt must have one extra space in front'
                human_segmentation_proportion = human_segmentation_proportion[1:]
        assert len(human_segmentation_proportion) == len(original_txt.split(' '))
        segmentation_prop_smoothed = gaussian_filter1d(human_segmentation_proportion,2.5)
        
        word_indices_in_tokenized_txt = find_word_indices_in_tokenized_txt(original_txt,tokenized_txt,model_initial_char)
        story_word_indices_in_tokenized_txt[story]=np.array(word_indices_in_tokenized_txt)
        story_human_segmentation_prop[story] = human_segmentation_proportion
        story_human_segmentation_prop_smoothed[story] = segmentation_prop_smoothed
        story_subject_segmentation_bool[story] = load_subject_segmentation_bool(segmentation_dir,story,original_txt)
    return story_word_indices_in_tokenized_txt,story_human_segmentation_prop,story_human_segmentation_prop_smoothed,story_subject_segmentation_bool

def load_subject_segmentation_bool(segmentation_dir,story,original_txt):
    segmentation_df = pd.read_csv(os.path.join(segmentation_dir,story,story+'_segmentation.csv'))
    segmentation_bool = []
    for subject in np.arange(len(segmentation_df)):
        #print(subject)
        subject_id = segmentation_df['subject'][subject]
        subject_segmentation = segmentation_df['segmented_txt'][subject]
        if subject_segmentation[0] == '\n':
            subject_segmentation = subject_segmentation[1:] 
        subject_segmentation_split = subject_segmentation.split('\n')
        subject_segmentation_wordlist = segmentation_to_word_list(subject_segmentation_split)
        if len(subject_segmentation_wordlist)!=len(original_txt.split(' ')):
            print('aligning ',subject_id,story)
            assert len(subject_segmentation_wordlist)-len(original_txt.split(' '))==-1, 'should be missing one empty string'
            for i,w in enumerate(subject_segmentation_wordlist):
                if subject_segmentation_wordlist[i]!='' and original_txt.split(' ')[i]=='':
                    index = i
                    #print(i,w,original_txt.split(' ')[i])
                    break
            subject_segmentation_wordlist.insert(index, '') # add the missing space
        for i,w in enumerate(subject_segmentation_wordlist):
            if subject_segmentation_wordlist[i].strip('\n')!=original_txt.split(' ')[i]:
                #index = i
                print(i,w,original_txt.split(' ')[i])
        assert len(subject_segmentation_wordlist)==len(original_txt.split(' ')),'length should match'
        subject_segmentation_bool = ['\n' in w for w in subject_segmentation_wordlist]
        segmentation_bool.append(subject_segmentation_bool)
    return np.array(segmentation_bool)

def raw_stimulus(stories,story_word_indices_in_tokenized_txt,story_human_segmentation_prop_smoothed,moth_output_dir,story_to_idx_mapping,feature,nfeat,story_subject_segmentation_bool):
    #fig,(ax1,ax2)=plt.subplots(2,1,figsize = (10,6))
    z_score_feat = True
    z_score_resp = False
    story_id_all = []
    sig_mat_all = []
    button_all =[]
    all_stories_sig = {}
    story_filtered_subject_segmentation_bool = {}
    for i,story in enumerate(stories):
        word_indices_in_tokenized_txt = story_word_indices_in_tokenized_txt[story]
        human_segmentation_proportion = story_human_segmentation_prop_smoothed[story]
        #print(word_indices_in_tokenized_txt[0])
        word_onset_token_idx = word_indices_in_tokenized_txt[:,0]
        if feature =='kl':
            story_kl = np.load(os.path.join(moth_output_dir,story,'kl.npy'))
            assert word_indices_in_tokenized_txt[-1][1]==story_kl.shape[0], 'last token in word_indices_in_tokenized_txt must match shape of story_kl'
            # originally the last one is word t-1 predicting t and t predicting nonexistent next word. So now this is from token position 1 to the end of story
            story_kl = story_kl[:-1] 
            story_kl = np.insert(story_kl,0,np.nan)
            feature_arr = story_kl
        elif feature =='ce':
            with open(os.path.join(moth_output_dir,story,'cross_entropy.pkl'),'rb') as f:
                original_ce = pickle.load(f)
            original_ce = original_ce.cpu().numpy()
            assert word_indices_in_tokenized_txt[-1][1]==original_ce.shape[0], 'last token in word_indices_in_tokenized_txt must match shape of story_kl'
            feature_arr = original_ce
        
        feature_at_word_onsets = feature_arr[word_onset_token_idx]
        assert feature_at_word_onsets.shape[0]==human_segmentation_proportion.shape[0]
        if feature =='kl':
            nan_indices = np.isnan(feature_at_word_onsets)
            feature_at_word_onsets = feature_at_word_onsets[~nan_indices]
            human_segmentation_proportion = human_segmentation_proportion[~nan_indices]

        if z_score_feat:
            zscore_feature_at_word_onsets = z_score(feature_at_word_onsets)
            sig_mat = build_signal_mat2(zscore_feature_at_word_onsets,nfeat)
            all_stories_sig[story] = zscore_feature_at_word_onsets
        else:
            sig_mat = build_signal_mat2(feature_at_word_onsets,nfeat)
            all_stories_sig[story] =feature_at_word_onsets
        sig_mat=sig_mat[nfeat:,:] # the first nfeat is empty backwards
        story_id_all.extend([story_to_idx_mapping[story]]*len(sig_mat))
        if i==0:
            sig_mat_all=sig_mat
        else:
            sig_mat_all=np.vstack([sig_mat_all,sig_mat])
        trun_human_segmentation_proportion = human_segmentation_proportion[nfeat:]
        if z_score_resp:
            zscore_human_segmentation_proportion = (trun_human_segmentation_proportion-np.mean(trun_human_segmentation_proportion))/np.std(trun_human_segmentation_proportion)
            button_all.append(zscore_human_segmentation_proportion)
        else:
            trun_human_segmentation_proportion = trun_human_segmentation_proportion/np.max(trun_human_segmentation_proportion)
            button_all.append(trun_human_segmentation_proportion)
        
        if story_subject_segmentation_bool is not None:
            subject_segmentation_bool = story_subject_segmentation_bool[story]
            filtered_subject_segmentation_bool = []
            for s_bool in subject_segmentation_bool:
                s_bool[-1] = False # set last word to false because it's technically not a segmentation
                if feature =='kl':
                    s_bool = s_bool[~nan_indices] 
                s_bool = s_bool[nfeat:]
                assert len(s_bool) == len(sig_mat)
                filtered_subject_segmentation_bool.append(s_bool)
            story_filtered_subject_segmentation_bool[story_to_idx_mapping[story]] = filtered_subject_segmentation_bool
    
        #print(sig_mat.shape)
    #     ax1.plot(story_kl_at_word_onsets,alpha = 0.5)
    #     ax2.plot(zscore_story_kl_at_word_onsets,alpha = 0.5)
    button_all = np.concat(button_all)
    assert button_all.shape[0]==sig_mat_all.shape[0]
    story_id_all = np.array(story_id_all)
    return sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool

def punctuated_stimulus(stories,story_human_segmentation_prop_smoothed,all_stories_kl_at_word_onsets_punctuated,all_stories_word_indices_in_tokenized_txt_punctuated,punctuated_output_dir,original_transcript_dir,timing_dir,story_to_idx_mapping,feature,nfeat,story_subject_segmentation_bool):
    z_score_feat = True
    z_score_resp = False
    story_id_all = []
    sig_mat_all = []
    button_all =[]
    all_stories_sig = {}
    story_filtered_subject_segmentation_bool = {} # key is story index, value is list of subject segmentation bool arrays aligned to sig_mat_all (after getting rid of leading nans and first nfeat indices)
    for i,story in enumerate(stories):
        # these two arrays should have same lengths for most story, but for souls there are two extra spaces in original_txt
        human_segmentation_proportion = story_human_segmentation_prop_smoothed[story] # this has same length as original_txt.split(' ')
        if story_subject_segmentation_bool is not None:
            subject_segmentation_bool = story_subject_segmentation_bool[story] # this has same length as original_txt.split(' ')
        if feature =='kl_punctuated_kumar':
            story_kl_at_word_onsets_punctuated = all_stories_kl_at_word_onsets_punctuated[story] # this has same length as timing_df
        #print(story,human_segmentation_proportion.shape,story_kl_at_word_onsets_punctuated.shape)
        elif feature =='kl_punctuated':
            word_indices_in_tokenized_txt = all_stories_word_indices_in_tokenized_txt_punctuated[story]
            word_onset_token_idx = word_indices_in_tokenized_txt[:,0]
            story_kl = np.load(os.path.join(punctuated_output_dir,story,'kl.npy')) # punctuated!
            # originally the last one is word t-1 predicting t and t predicting nonexistent next word. So now this is from token position 1 to the end of story
            story_kl = story_kl[:-1] 
            story_kl = np.insert(story_kl,0,np.nan)
            story_kl_at_word_onsets_punctuated = story_kl[word_onset_token_idx]
        # deal with spaces in 'soul'
        with open(os.path.join(original_transcript_dir,'%s.txt'%story),'r') as f:
            original_txt = f.read()
        timing_df = pd.read_csv(os.path.join(timing_dir,'%s_timing.csv'%story))
        if len(timing_df) != len(original_txt.split(' ')):
            empty_string_idx = []
            for i in range(len(original_txt.split(' '))):
                if original_txt.split(' ')[i] =='':
                    print(story,i)
                    empty_string_idx.append(i)
            human_segmentation_proportion = np.delete(human_segmentation_proportion,empty_string_idx)
        assert story_kl_at_word_onsets_punctuated.shape[0]==human_segmentation_proportion.shape[0]
        
        if story_subject_segmentation_bool is not None and len(timing_df) != len(original_txt.split(' ')):
            cleaned_subject_segmentation_bool = []
            for s_bool in subject_segmentation_bool:
                    s_bool = np.delete(s_bool,empty_string_idx)
                    assert story_kl_at_word_onsets_punctuated.shape[0]==s_bool.shape[0]
                    cleaned_subject_segmentation_bool.append(s_bool)
            subject_segmentation_bool = np.array(cleaned_subject_segmentation_bool)

        
        # get rid of leading nan in kl
        nan_indices = np.isnan(story_kl_at_word_onsets_punctuated) # first index is nan
        story_kl_at_word_onsets_punctuated = story_kl_at_word_onsets_punctuated[~nan_indices]
        human_segmentation_proportion = human_segmentation_proportion[~nan_indices]
        
        if z_score_feat:
            zscore_story_kl_at_word_onsets = z_score(story_kl_at_word_onsets_punctuated)
            all_stories_sig[story] = zscore_story_kl_at_word_onsets
            sig_mat = build_signal_mat2(zscore_story_kl_at_word_onsets,nfeat)
        else:
            sig_mat = build_signal_mat2(story_kl_at_word_onsets_punctuated,nfeat)
            all_stories_sig[story] = story_kl_at_word_onsets_punctuated
        sig_mat=sig_mat[nfeat:,:] # the first nfeat is empty backwards
        story_id_all.extend([story_to_idx_mapping[story]]*len(sig_mat))
        if i==0:
            sig_mat_all=sig_mat
        else:
            sig_mat_all=np.vstack([sig_mat_all,sig_mat])
        trun_human_segmentation_proportion = human_segmentation_proportion[nfeat:]
        if z_score_resp:
            zscore_human_segmentation_proportion = (trun_human_segmentation_proportion-np.mean(trun_human_segmentation_proportion))/np.std(trun_human_segmentation_proportion)
            button_all.append(zscore_human_segmentation_proportion)
        else:
            trun_human_segmentation_proportion = trun_human_segmentation_proportion/np.max(trun_human_segmentation_proportion)
            button_all.append(trun_human_segmentation_proportion)
        if story_subject_segmentation_bool is not None:
            filtered_subject_segmentation_bool = []
            for s_bool in subject_segmentation_bool:
                s_bool[-1] = False # set last word to false because it's technically not a segmentation
                s_bool = s_bool[~nan_indices] # get rid of the first index which is nan in kl
                s_bool = s_bool[nfeat:] # align with sig_mat which also gets rid of first nfeat indices
                assert len(s_bool) == len(sig_mat)
                filtered_subject_segmentation_bool.append(s_bool)
            story_filtered_subject_segmentation_bool[story_to_idx_mapping[story]] = filtered_subject_segmentation_bool
        #print(sig_mat.shape)
    #     ax1.plot(story_kl_at_word_onsets,alpha = 0.5)
    #     ax2.plot(zscore_story_kl_at_word_onsets,alpha = 0.5)
    button_all = np.concat(button_all)
    assert button_all.shape[0]==sig_mat_all.shape[0]
    story_id_all = np.array(story_id_all)
    return sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool

def get_all_stories_punctuation_bool(stories, punctuated_transcript_dir, original_transcript_dir, timing_dir):
    """Return a dict mapping each story to a binary punctuation mask aligned to the transcript words."""
    all_stories_punctuation_bool = {}
    for story in stories:
        with open(os.path.join(punctuated_transcript_dir, '%s_punctuated.txt' % story), 'r') as f:
            punctuated_txt = f.read()
        with open(os.path.join(original_transcript_dir, '%s.txt' % story), 'r') as f:
            original_txt = f.read()
        timing_df = pd.read_csv(os.path.join(timing_dir, '%s_timing.csv' % story))
        punctuated_txt_split = re.split(r"[ -]+", punctuated_txt)
        original_txt_split = re.split(r"[ -]+", original_txt)

        assert len(punctuated_txt_split) == len(original_txt_split), (
            'word split of punctuated and original txt must have the same length'
        )
        strip_chars = string.punctuation + '-–“”’'
        for i in range(len(punctuated_txt_split)):
            punc_word = punctuated_txt_split[i]
            original_word = original_txt_split[i]
            punc_word_strip = punc_word.lower().translate(str.maketrans('', '', strip_chars))
            original_word_strip = original_word.lower().translate(str.maketrans('', '', strip_chars))
            if punc_word_strip != original_word_strip:
                print('word between punctuated vs. original is not the same:', story, i, punc_word, original_word)

        if len(original_txt.split(' ')) != len(original_txt_split):
            empty_string_indices = [i for i, w in enumerate(original_txt.split(' ')) if w == '']
            for i in empty_string_indices:
                original_txt_split.insert(i, '')
                punctuated_txt_split.insert(i, '')
            assert original_txt_split == original_txt.split(' '), 'two splits must be the same after insertion'

        punctuation_indices = np.array([i for i, w in enumerate(punctuated_txt_split) if any(c in w for c in '.!?')])
        punctuation_bool = np.zeros(len(original_txt.split(' ')))
        punctuation_bool[punctuation_indices] = 1
        all_stories_punctuation_bool[story] = punctuation_bool
    return all_stories_punctuation_bool


def punctuation_only(stories,story_word_indices_in_tokenized_txt,story_human_segmentation_prop_smoothed,moth_output_dir,story_to_idx_mapping,feature,nfeat,story_subject_segmentation_bool,punctuated_transcript_dir,original_transcript_dir,timing_dir,all_stories_word_indices_in_tokenized_txt_punctuated):
    z_score_feat = True
    z_score_resp = False
    all_stories_punctuation_bool = get_all_stories_punctuation_bool(
        stories,
        punctuated_transcript_dir,
        original_transcript_dir,
        timing_dir,
    )

    story_id_all = []
    sig_mat_all = []
    button_all =[]
    all_stories_sig = {}
    story_filtered_subject_segmentation_bool = {} # key is story index, value is list of subject segmentation bool arrays aligned to sig_mat_all (after getting rid of leading nans and first nfeat indices)
    for i,story in enumerate(stories):
        # these two arrays should have same lengths for most story, but for souls there are two extra spaces in original_txt
        human_segmentation_proportion = story_human_segmentation_prop_smoothed[story] # this has same length as original_txt.split(' ')
        story_punctuation_bool = all_stories_punctuation_bool[story]
        #print(story_punctuation_bool.shape[0],human_segmentation_proportion.shape[0])
        assert story_punctuation_bool.shape[0]==human_segmentation_proportion.shape[0]
        
        if z_score_feat:
            zscore_story_punctuation_bool = z_score(story_punctuation_bool)
            sig_mat = build_signal_mat2(zscore_story_punctuation_bool,nfeat)
            all_stories_sig[story] = zscore_story_punctuation_bool
        else:
            sig_mat = build_signal_mat2(story_punctuation_bool,nfeat)
            all_stories_sig[story] = story_punctuation_bool
        sig_mat=sig_mat[nfeat:,:] # the first nfeat is empty backwards
        story_id_all.extend([story_to_idx_mapping[story]]*len(sig_mat))
        if i==0:
            sig_mat_all=sig_mat
        else:
            sig_mat_all=np.vstack([sig_mat_all,sig_mat])
        trun_human_segmentation_proportion = human_segmentation_proportion[nfeat:]
        if z_score_resp:
            zscore_human_segmentation_proportion = (trun_human_segmentation_proportion-np.mean(trun_human_segmentation_proportion))/np.std(trun_human_segmentation_proportion)
            button_all.append(zscore_human_segmentation_proportion)
        else:
            trun_human_segmentation_proportion = trun_human_segmentation_proportion/np.max(trun_human_segmentation_proportion)
            button_all.append(trun_human_segmentation_proportion)
        if story_subject_segmentation_bool is not None:
            subject_segmentation_bool = story_subject_segmentation_bool[story]
            filtered_subject_segmentation_bool = []
            for s_bool in subject_segmentation_bool:
                s_bool[-1] = False # set last word to false because it's technically not a segmentation
                s_bool = s_bool[nfeat:]
                assert len(s_bool) == len(sig_mat)
                filtered_subject_segmentation_bool.append(s_bool)
            story_filtered_subject_segmentation_bool[story_to_idx_mapping[story]] = filtered_subject_segmentation_bool

    button_all = np.concat(button_all)
    assert button_all.shape[0]==sig_mat_all.shape[0]
    story_id_all = np.array(story_id_all)
    return sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool

def punctuation_kl_variance_partitioning(stories,nfeat,punctuated_transcript_dir,original_transcript_dir,timing_dir,story_human_segmentation_prop_smoothed,all_stories_kl_at_word_onsets_punctuated,story_to_idx_mapping,args):
    # punctuated KL, (skipping punctuation like Kumar et al.)
    z_score_feat = True
    z_score_resp = False
    story_id_all = []
    sig_mat_all_kl = []
    sig_mat_all_punc = []
    button_all =[]
    all_stories_punctuation_bool = get_all_stories_punctuation_bool(stories, punctuated_transcript_dir, original_transcript_dir, timing_dir)
    for i,story in enumerate(stories):    
        # these two arrays should have same lengths for most story, but for souls there are two extra spaces in original_txt
        human_segmentation_proportion = story_human_segmentation_prop_smoothed[story] # this has same length as original_txt.split(' ')
        story_kl_at_word_onsets_punctuated = all_stories_kl_at_word_onsets_punctuated[story] # this has same length as timing_df
        #print(story,human_segmentation_proportion.shape,story_kl_at_word_onsets_punctuated.shape)
        story_punctuation_bool = all_stories_punctuation_bool[story]
        assert story_punctuation_bool.shape[0]==human_segmentation_proportion.shape[0]

        # deal with spaces in 'soul'
        with open(os.path.join(original_transcript_dir,'%s.txt'%story),'r') as f:
            original_txt = f.read()
        timing_df = pd.read_csv(os.path.join(timing_dir,'%s_timing.csv'%story))
        if len(timing_df) != len(original_txt.split(' ')):
            empty_string_idx = []
            for i in range(len(original_txt.split(' '))):
                if original_txt.split(' ')[i] =='':
                    print(story,i)
                    empty_string_idx.append(i)
            human_segmentation_proportion = np.delete(human_segmentation_proportion,empty_string_idx)     
            story_punctuation_bool = np.delete(story_punctuation_bool,empty_string_idx)
        assert story_kl_at_word_onsets_punctuated.shape[0]==human_segmentation_proportion.shape[0]==story_punctuation_bool.shape[0]
        
        # get rid of leading nan in kl
        nan_indices = np.isnan(story_kl_at_word_onsets_punctuated) # first index is nan
        story_kl_at_word_onsets_punctuated = story_kl_at_word_onsets_punctuated[~nan_indices]
        human_segmentation_proportion = human_segmentation_proportion[~nan_indices]
        story_punctuation_bool = story_punctuation_bool[~nan_indices]
        
        if z_score_feat:
            zscore_story_kl_at_word_onsets = z_score(story_kl_at_word_onsets_punctuated)
            zscore_story_punctuation_bool = z_score(story_punctuation_bool)
            sig_mat_kl = build_signal_mat2(zscore_story_kl_at_word_onsets,nfeat)
            sig_mat_punctuation = build_signal_mat2(zscore_story_punctuation_bool,nfeat)
        else:
            sig_mat_kl = build_signal_mat2(story_kl_at_word_onsets_punctuated,nfeat)
            sig_mat_punctuation = build_signal_mat2(story_punctuation_bool,nfeat)
        
        sig_mat_kl=sig_mat_kl[nfeat:,:] # the first nfeat is empty backwards
        sig_mat_punctuation = sig_mat_punctuation[nfeat:,:]
        assert sig_mat_kl.shape==sig_mat_punctuation.shape
        story_id_all.extend([story_to_idx_mapping[story]]*len(sig_mat_kl))
        if i==0:
            sig_mat_all_kl=sig_mat_kl
            sig_mat_all_punc = sig_mat_punctuation
        else:
            sig_mat_all_kl=np.vstack([sig_mat_all_kl,sig_mat_kl])
            sig_mat_all_punc = np.vstack([sig_mat_all_punc,sig_mat_punctuation])
        trun_human_segmentation_proportion = human_segmentation_proportion[nfeat:]
        if z_score_resp:
            zscore_human_segmentation_proportion = (trun_human_segmentation_proportion-np.mean(trun_human_segmentation_proportion))/np.std(trun_human_segmentation_proportion)
            button_all.append(zscore_human_segmentation_proportion)
        else:
            trun_human_segmentation_proportion = trun_human_segmentation_proportion/np.max(trun_human_segmentation_proportion)
            button_all.append(trun_human_segmentation_proportion)
        #print(sig_mat.shape)
    #     ax1.plot(story_kl_at_word_onsets,alpha = 0.5)
    #     ax2.plot(zscore_story_kl_at_word_onsets,alpha = 0.5)
    button_all = np.concat(button_all)
    sig_mat_all_joint = np.hstack([sig_mat_all_punc,sig_mat_all_kl])
    assert button_all.shape[0]==sig_mat_all_punc.shape[0]==sig_mat_all_kl.shape[0]==sig_mat_all_joint.shape[0]
    story_id_all = np.array(story_id_all)

    r2_kl, betas_kl,result_kl,cvres_kl,best_models_kl =compute_grid_search_regression(sig_mat_all_kl,story_id_all, button_all, args)
    r2_punc, betas_punc,result_punc,cvres_punc,best_models_punc =compute_grid_search_regression(sig_mat_all_punc,story_id_all, button_all, args)
    r2_joint, betas_joint,result_joint,cvres_joint,best_models_joint =compute_grid_search_regression(sig_mat_all_joint,story_id_all, button_all, args)

    r2_punc = np.array(r2_punc)
    r2_kl = np.array(r2_kl)
    r2_joint = np.array(r2_joint)

    d_joint_punc = r2_joint - r2_punc
    d_joint_kl = r2_joint - r2_kl
    pval_punc = sign_flip_test(d_joint_punc)
    pval_kl = sign_flip_test(d_joint_kl)
    return pval_punc, pval_kl, d_joint_punc, d_joint_kl, r2_joint, r2_punc, r2_kl

def sign_flip_test(d):
    # Sign-flip (exact permutation) test for a paired difference.
    # Tests whether the mean difference d_i = R2_full_i - R2_reduced_i across
    # leave-one-story-out folds is reliably different from zero, under the null
    # that each story's difference could equally well have been positive or
    # negative (i.e., no true effect of the added predictor).
    # Enumerates all 2^n possible sign assignments (exact for small n), computes
    # the mean difference under each, and compares the observed mean to this
    # null distribution to get an exact two-sided p-value.
    n = len(d)
    observed_mean = np.mean(d)

    # Enumerate all 2^n sign-flip patterns exactly (feasible for n=8: 256 total)
    null_means = []
    for signs in product([1, -1], repeat=n):
        flipped = d * np.array(signs)
        null_means.append(np.mean(flipped))

    null_means = np.array(null_means)

    # Two-sided p-value: how extreme is the observed mean vs. the null distribution?
    p_value = np.mean(np.abs(null_means) >= np.abs(observed_mean))
    return p_value

def main(args):
    model_name = args.model
    save_dir = args.save_dir
    model_save_dir_name = model_to_path_dict[model_name]['save_dir_name']
    model_initial_char = model_to_path_dict[model_name]['initial_char']

    moth_output_dir = os.path.join(save_dir,model_save_dir_name,'moth_stories_output')
    original_transcript_dir = args.original_transcript_dir
    segmentation_dir = args.segmentation_dir
    timing_dir = args.timing_dir
    regression_save_dir = os.path.join(save_dir,model_save_dir_name,'segmentation_regression')
    if not os.path.exists(regression_save_dir):
        os.makedirs(regression_save_dir)

    stories = ['pieman','alternateithicatom','odetostepfather','souls','legacy','wheretheressmoke','adventuresinsayingyes','inamoment']
    story_to_idx_mapping = {s:stories.index(s)+1 for s in stories}
    story_word_indices_in_tokenized_txt,story_human_segmentation_prop,story_human_segmentation_prop_smoothed,story_subject_segmentation_bool = get_human_segmentation_prop(stories,moth_output_dir,segmentation_dir,original_transcript_dir,model_initial_char,by_subject = args.roc)
    if args.punctuation_kl_var_partition:
        kl_regression_dir = os.path.join(save_dir,model_save_dir_name,'kl_regression')
        with open(os.path.join(kl_regression_dir,'regression_dict.pkl'),'rb') as f:
            regression_dict = pickle.load(f)
        all_stories_kl_at_word_onsets_punctuated = regression_dict['kl_at_word_onsets_punctuated']
        pval_punc, pval_kl, d_joint_punc, d_joint_kl, r2_joint, r2_punc, r2_kl = punctuation_kl_variance_partitioning(stories,args.nfeat,args.punctuated_transcript_dir,original_transcript_dir,timing_dir,story_human_segmentation_prop_smoothed,all_stories_kl_at_word_onsets_punctuated,story_to_idx_mapping,args)
        var_par_dir = os.path.join(regression_save_dir,'punctuated_kl_variance_partitioning')
        if not os.path.exists(var_par_dir):
            os.makedirs(var_par_dir)
        var_par_results = {'pval_punc':pval_punc,'pval_kl':pval_kl,'d_joint_punc':d_joint_punc,'d_joint_kl':d_joint_kl,'r2_joint':r2_joint,'r2_punc':r2_punc,'r2_kl':r2_kl}
        with open(os.path.join(var_par_dir,'punctuated_kl_variance_partitioning_results_%i_%s_%s.pkl'%(args.nfeat,args.reg_model,args.stat)),'wb') as f:
            pickle.dump(var_par_results,f)
    elif args.feature == 'punctuation_only':
        kl_regression_dir = os.path.join(save_dir,model_save_dir_name,'kl_regression')
        with open(os.path.join(kl_regression_dir,'regression_dict.pkl'),'rb') as f:
            regression_dict = pickle.load(f)
        all_stories_word_indices_in_tokenized_txt_punctuated = regression_dict['word_indices_in_tokenized_txt_punctuated']
        sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool = punctuation_only(stories,story_word_indices_in_tokenized_txt,story_human_segmentation_prop_smoothed,moth_output_dir,story_to_idx_mapping,args.feature,args.nfeat,story_subject_segmentation_bool,args.punctuated_transcript_dir,original_transcript_dir,timing_dir,all_stories_word_indices_in_tokenized_txt_punctuated)
    elif 'punc' not in args.feature:
        sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool = raw_stimulus(stories,story_word_indices_in_tokenized_txt,story_human_segmentation_prop_smoothed,moth_output_dir,story_to_idx_mapping,args.feature,args.nfeat,story_subject_segmentation_bool)
    else:
        punctuated_output_dir = os.path.join(save_dir,model_save_dir_name,'moth_stories_output_punctuated')
        kl_regression_dir = os.path.join(save_dir,model_save_dir_name,'kl_regression')
        with open(os.path.join(kl_regression_dir,'regression_dict.pkl'),'rb') as f:
            regression_dict = pickle.load(f)
        all_stories_kl_at_word_onsets_punctuated = regression_dict['kl_at_word_onsets_punctuated']
        all_stories_word_indices_in_tokenized_txt_punctuated = regression_dict['word_indices_in_tokenized_txt_punctuated']
        sig_mat_all,button_all,story_id_all,all_stories_sig,story_filtered_subject_segmentation_bool = punctuated_stimulus(stories,story_human_segmentation_prop_smoothed,all_stories_kl_at_word_onsets_punctuated,all_stories_word_indices_in_tokenized_txt_punctuated,punctuated_output_dir,original_transcript_dir,timing_dir,story_to_idx_mapping,args.feature,args.nfeat,story_subject_segmentation_bool)
    
    if args.roc:
        corr_score_ridge, betas_ridge,result_ridge,cvres_ridge,best_models,test_story_auc,test_story_pval,test_story_tpr_fpr =compute_grid_search_regression_individual_subject_test(sig_mat_all,story_id_all, button_all, args,story_filtered_subject_segmentation_bool,n_perm = 1000,threshold = args.threshold)
        roc_result = {'test_story_auc':test_story_auc,'test_story_pval':test_story_pval,'test_story_tpr_fpr':test_story_tpr_fpr}
        with open(os.path.join(regression_save_dir,'roc_result_%s_%i_%s_%s.pkl'%(args.feature,args.nfeat,args.reg_model,args.stat)),'wb') as f:
            pickle.dump(roc_result,f)

    if args.perm:
        corr_score, betas,result,cvres,best_models =compute_grid_search_regression(sig_mat_all,story_id_all, button_all, args)
        print(corr_score)
        mean_sig_length = np.max([all_stories_sig[story].shape[0] for story in stories])
        nPerm = int(mean_sig_length)
        print('nPerm',nPerm)
        #nPerm=4000
        lin_s = list(map(int, np.linspace(-nPerm, nPerm, 2*nPerm+1)))
        lin_s.remove(0)
        corr_perm_all, betas_perm_all = _run_permutation_regressions(
            lin_s,
            stories,
            all_stories_sig,
            story_id_all,
            button_all,
            args,
            use_parallel=args.parallel_perm,
            n_jobs=args.perm_n_jobs if args.perm_n_jobs is not None else (os.cpu_count() or 1),
        )
        
        out_file = os.path.join(regression_save_dir,'perm_regression_outputs_%s_%i_%s_%s'%(args.feature,args.nfeat,args.reg_model,args.stat))
        np.savez(out_file,betas_perm_all=betas_perm_all, betas=betas, corr_score=corr_score,corr_perm_all=corr_perm_all)
        pval=stats.percentileofscore(np.nanmean(corr_perm_all,axis=1),np.mean(corr_score))
        log_file = os.path.join(regression_save_dir,'correlation_pval%s_%i_%s_%s'%(args.feature,args.nfeat,args.reg_model,args.stat))
        
        f = open(log_file, 'w')
        f.write('Feature = %s, Nfeatures= %i,  %s Mean Corr = %.3f, p-val= %.3f' % (args.feature, args.nfeat,args.stat,np.mean(corr_score),pval))
        print(pval)
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--save_dir",default='../generated')
    parser.add_argument("--segmentation_dir",default = '../behavior_data/segmentation')
    parser.add_argument("--original_transcript_dir",default = "../behavior_data/transcripts/moth_stories",help = "directory storing lower case transcripts of story")
    parser.add_argument("--exclusion_dir",default = "../behavior_data/exclusion",help = "directory storing exclusion dfs")
    parser.add_argument("--event_timing_dir",default = '../behavior_data/event_timing',help = "directory storing event timing dfs")
    parser.add_argument("--timing_dir",default = '../behavior_data/transcripts/timing',help="directory storing timing of each word in transcript")
    parser.add_argument("--punctuated_transcript_dir",default='../behavior_data/transcripts/punctuated',help="directory storing punctuated transcripts")
    parser.add_argument("--punctuated",action ='store_true',help ='add punctuated results')
    parser.add_argument("--model",default = 'Llama3-8b-instruct')
    parser.add_argument("--feature",type = str,choices = ['ce','kl','kl_punctuated','kl_punctuated_kumar','punctuation_only'])
    parser.add_argument("--nfeat",type = int, default = 10)
    parser.add_argument("--stat",type = str,default ='Pearson',choices = ['Pearson','r2'])
    parser.add_argument("--reg_model",type = str,default = 'Lasso')
    parser.add_argument("--perm",action = 'store_true',help = 'whether to run permutation regression for statistical significance')
    parser.add_argument("--parallel_perm",action = 'store_true',help = 'run permutation regressions with threaded parallelism')
    parser.add_argument("--perm_n_jobs",type = int,default = None,help = 'number of worker threads to use when --parallel_perm is set')
    parser.add_argument("--roc",action = 'store_true',help = 'whether to calculate AUC for ROC curve')
    parser.add_argument("--threshold",type = float, default = 0.1, help = 'threshold for determining block size in permutation')
    parser.add_argument("--punctuation_kl_var_partition",action = 'store_true',help = 'run variance partitioning on punctuation only model and kl (punctuated version)')
    args = parser.parse_args()
    main(args)