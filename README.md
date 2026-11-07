# CRUISE
This repository contains code used in the paper "Efficient uniform sampling explains non-uniform memory of narrative stories" by Jianing Mu, Alison R. Preston and Alexander G. Huth. 

## Software requirements
This repo has been tested on Linux Ubuntu 24.04.2 with Python 3.9.7 and 3.10.8
Dependencies: pytorch (tested on 2.2.1 and 2.3.1), transformers (tested on 4.33.3 and 4.45.2), numpy (tested on 1.24.3, 1.26.4), scipy (tested on 1.10.1, 1.13.0), pandas (tested on 1.5.0, 1.5.2), matplotlib, seaborn, statsmodels, scikit-learn, natsort, joblib. 
All code involving LLM inference was ran on a compute node with three 40G A100 GPUs. To demonstrate the subsequent analyses for users without GPU access, inference results are provided. 
See ```notebooks``` for demos and replications of figures in the main text. Runtime for all demos should be under a minute, except for ```recall_generation_demo.ipynb```, which takes about 20s per recall. 

## Setup
Download and unzip the data from box. Within the same directory, clone the current repo. The final directory structure should be: 
```
- root dir
    - CRUISE
    - ablation 
    - behavior_data
    - generated
    - exp2 (Experiment 2 data)
        - behavior_data
        - generated
```
Bash scripts in ```bash_files``` use relative paths and should be run from within ```CRUISE/bash_files```. Python scripts called directly (e.g. ```perm_regression.py```, ```roc_analysis.py```) should be run from within ```CRUISE```. 
Intermediate data (inference outputs, segmentation, etc.) is provided so that all analyses downstream of LLM inference can be run.

## Split the story into windows of equal durations, number of windows = number of events (Fig.2 uniform encoding, and supplemental boundary analysis)
1. First run ```run_split_story_by_even_duration.sh``` locally to generate the unadjusted splits of the story. Output is under ```behavior_data/story_split_timing```. Then manually adjust for phrase boundaries. 
2. (Requires a GPU cluster. Output data is provided to demo subsequent steps.) Run ```run_story_even_split_analysis.sh```, packages inference code, runs both instruct and non-instruct concatenations to get I(Xi;R) (run_recall_explained_events.sh). Also calculates H(X) (get_logits), I(Xi;Xj) (run_pairwise_events.sh). Inference scripts called in this bash file uses --split_story_by_duration to indicate the even duration condition. 
3. Use ```run_analyze_uniform_encoding.sh``` to calculate CRUISE and other models. Generates dataframes for plotting. Also see demo in ```uniform encoding hypothesis-split story evenly by duration.ipynb```. Positional arguments: model, random_recalls, event, align_offset, split_by_token, factor, calibration_temp, sr_exp, stories. For this analysis: ```bash run_analyze_uniform_encoding.sh "Llama3-8b-instruct"``` (all other arguments default to false/None, and stories default to all 8 stories)
4. Use ```uniform encoding hypothesis combine stories-split story evenly by duration.ipynb``` to generate scatter plots with regression fits
5. Use ```uniform encoding hypothesis - split story evenly by duration - compare models.ipynb``` to generate bar plots of R^2
6. Use ```Uniform encoding hypothesis - by subject prevalence-split story evenly - cleaned.ipynb``` to perform subject-level significance testing 


## Boundary analysis that splits the story into equal-duration or equal-token windows (Fig.3, 5, and supplemental results)
1. Split into equal token with 1.5xnumber of events 
    1. Generate windows ```split_story_by_tokens.py --story {story} --factor 1.5``` Outputs 'story_even_token_factor_%.1f.csv'%args.factor in behavior_data/story_split_timing
    2. Adjust for phrase boundaries manually, save them as 'story_even_token_factor_%.1f_adjusted.csv'%args.factor, send them back to the cluster
    3. (Requires a GPU cluster. Output data is provided to demo subsequent steps.) Run ```bash run_story_even_split_analysis.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "true" "false" 1.5 "true"``` This calls ```run_split_story_by_even_duration.sh``` to align the adjusted windows to the correct timing and tokens and recalculate whether each window is a boundary or not, then run the full LLM inference. Results are under pairwise_event/{story}/'story_split_tokens_factor_%.1f_adjusted'%args.factor. (default factor 1.5, meaning the number of windows = 1.5 times the number of events)
    4. Calculate CRUISE, surprisal weighted sampling and controls using ```uniform encoding hypothesis-split story evenly by tokens-split with factor.ipynb```, or with ```bash run_analyze_uniform_encoding.sh "Llama3-8b-instruct" "false" "false" "false" "true" 1.5```. Outputs ```uniform_encoding_df_even_split_by_tokens_factor_1.5_adjusted.csv``` in ```generated/{model}/uniform_encoding/{story}```
    5. Plot using ```split story by tokens - cleaned for plotting.ipynb```
2. Split into equal duration with 1.5 x number of events (Supplemental Fig. S10)
    1. Generate windows ```bash run_split_story_by_even_duration.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "false" 1.5 "false"```. Outputs 'story_even_duration_factor_%.1f.csv'%args.factor in behavior_data/story_split_timing
    2. Adjust for phrase boundaries manually, save them as 'story_even_duration_factor_%.1f_adjusted.csv'%args.factor
    3. Run ```bash run_story_even_split_analysis.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "true" "false" 1.5 "false"``` This calls run_split_story_by_even_duration.sh to align the adjusted windows to the correct timing and tokens, then run the full LLM inference. Results are under pairwise_event/{story}/'story_split_timing_factor_%.1f_adjusted'%args.factor
    4. Calculate CRUISE, surprisal weighted sampling and controls using code similar to the equal token split
    5. Plot using ```split story evenly by duration - chunks with boundary vs. no boundary cleaned for plotting.ipynb```

## Time courses of information properties around boundaries (Fig. 3fgh) and surprisal around boundaries vs. baseline (Supplement S22)
1. ```CE around event and punctuation boundaries.ipynb```: time courses of surprisal, relative entropy (KL), speech rate and information rate around event boundaries, with linear mixed-effects models comparing pre-boundary and post-boundary bins to the boundary bin. Also examines relative entropy around punctuations as a comparison. 
2. ```CE around event boundaries vs. random chunks.ipynb```: Comparing surprisal around boundaries to other positions in the story (Fig. S20), time courses of surprisal, speech rate and information rate at the word level (Fig. S22). 

## Predicting human segmentation from LLM information measures (Supplement & rebuttal)
1. ```boundary surprise analysis_cleaned.ipynb```: ROC of surprisal against human segmentation (block permutation test), and Wasserstein distance between thresholded-surprisal boundaries and human boundaries, in comparison to random and block-shuffled baselines (Fig. S21).
2.  ```roc_analysis_all_stories.ipynb``` plots per-subject ROC curves and p-values for all stories, and the regression results from ```perm_regression.py --roc```.
3. ```perm_regression.py```: lagged regression predicting human segmentation from LLM features based on Kumar et al. 2023 (```--feature``` one of ```ce```, ```kl```, ```kl_punctuated```, ```punctuation_only```; ```--nfeat``` number of lags), with leave-one-story-out cross-validation. ```--perm``` runs a circular-shift permutation test, ```--roc``` tests prediction of individual-subject segmentation, ```--punctuation_kl_var_partition``` runs variance partitioning between punctuation and KL. E.g. ```python perm_regression.py --feature kl --nfeat 10 --roc```. Results are saved in ```generated/{model}/segmentation_regression```. Uses the provided ```generated/{model}/kl_regression/regression_dict.pkl``` and ```moth_stories_output_punctuated``` inference results.
4. ```roc_analysis.py```: per-subject ROC and precision-recall AUC of LLM-predicted segmentation against each subject's segmentation, with block permutation tests. E.g. ```python roc_analysis.py --story pieman --instruct --exclusion```. Uses the provided ```generated/{model}/prolific_data/{story}_instruct/subject_segmentation_logits```.

## Mutual information of windows within event vs. across one event boundary (Fig. 4)
1. ```event_boundary_information.ipynb``` generates count balanced ablation stimuli in ```ablation/{model_name}/sliding_window_ablation/moth_stories```. 
2. (Requires a GPU cluster. Inference results are provided for subsequent analyses) Send stimuli to cluster for inference to obtain CE with ```sliding_ablation_entropy.py```. 
3. Analysis and replication of Fig. 4 in ```event_boundary_information_cleaned.ipynb```

## Experiment 2 (speech rate manipulation) analysis (Fig. 6)
Experiment 2 uses the same story windows as Experiment 1 (adjusted equal-token windows with factor 1.5, and adjusted equal-duration windows with number of windows = number of events), so story-only inference (H(X), I(Xi;Xj)) from Experiment 1 is reused and only recall inference is rerun on Experiment 2 recalls. Experiment 2 outputs are saved under ```exp2/generated```. Window timing for the shrink (pause-removal) condition is provided in ```generated/{model}/pairwise_event/{story}/{split}_shrunk```, and Sylber syllable-level results in ```generated/sylber```.

1. (Requires a GPU cluster. Output data is provided to demo subsequent steps.) Calculate I(Xi;R) for Experiment 2 recalls, for both base and instruct concatenations. From ```bash_files```:
    ```
    stories="pieman alternateithicatom odetostepfather legacy souls wheretheressmoke adventuresinsayingyes inamoment"
    # positional args: model, stories, adjusted, random_recalls, factor, split_by_token, calibration_temp, sr_exp
    # equal-token windows (factor 1.5)
    bash run_recall_explained_events_even_split.sh "Llama3-8b-instruct" "$stories" "true" "false" 1.5 "true" None "true"
    bash run_recall_explained_events_instruct_even_split.sh "Llama3-8b-instruct" "$stories" "true" "false" 1.5 "true" None "true"
    # equal-duration windows
    bash run_recall_explained_events_even_split.sh "Llama3-8b-instruct" "$stories" "true" "false" None "false" None "true"
    bash run_recall_explained_events_instruct_even_split.sh "Llama3-8b-instruct" "$stories" "true" "false" None "false" None "true"
    ```
2. Calculate CRUISE and other models to generate the uniform encoding dataframes. From ```bash_files```:
    ```
    # positional args: model, random_recalls, event, align_offset, split_by_token, factor, calibration_temp, sr_exp, stories
    # equal-token windows (factor 1.5)
    bash run_analyze_uniform_encoding.sh "Llama3-8b-instruct" "false" "false" "false" "true" 1.5 None "true" "$stories"
    # equal-duration windows
    bash run_analyze_uniform_encoding.sh "Llama3-8b-instruct" "false" "false" "false" "false" None None "true" "$stories"
    ```
    With ```sr_exp``` set to true, each call runs both the control (original) and shrink conditions, saving ```uniform_encoding_df_{split}_sr_original.csv``` and ```uniform_encoding_df_{split}_sr_shrink.csv``` in ```exp2/generated/{model}/uniform_encoding/{story}```. The notebook below also needs the Experiment 1 dataframes, generated by the same commands with ```sr_exp``` set to false. Repeat steps 1-2 for each model compared in the notebook (```Llama3-8b-instruct```, ```Llama3-8b```, ```Llama3.2-3b-instruct```, ```Llama3.2-3b-instruct_finetuned```, ```mistral-7b-instruct```, ```gemma-2-9b-it```).
3. ```speech rate experiment - compare models cleaned.ipynb```: computes participants' memory for shrink vs. no shrink conditions of experiment 2, and compares the predictive performance of CRUISE and other models. The models are fitted on Experiment 1 data and tested on experiment 2 data. Also generates the model universe plots for all LLMs and MI estimation methods 
4. ```compare segmentation between speech rate exp and original.ipynb```: compares event segmentation between Experiment 2 conditions and Experiment 1, including segmentation near pause-removal windows, and tests how much the change in segmentation explains the change in memory (Fig. S23). 

## Rate-distortion plot of LLM-generated recalls and human recalls (Fig. 7)
1. Generate recalls with attention temperature manipulation: On a cluster, ```generate_model_recall.py --story {story} --n 50 --temp 0.7 --att_to_story_start --prompt_number 1```. These are the parameters that all stories should have. Need to specify the desired attention temperature on line 131. 
    The attention temperature manipulation is implemented in an editable install of Transformers. See implementation of the attention temperature manipulation [here](https://github.com/mujn1461/private-transformers/blob/61e7edd0a1af2baa2447d9dbb2ffd85010581efc/src/transformers/models/llama/modeling_llama.py#L295). To install this version of Transformers, create a fresh conda environment. Install pytorch (tested with 2.0.1), numpy, pandas. Then clone the linked repository, run
    ```
    mv private-transformers transformers # renames the folder
    cd transformers
    git checkout llama-attention # make sure you're on the right branch
    pip install -e .
    ```
    Results are saved in csv files in ```generated/{model_name}/model_recall```. 
    If you rerun the ```generate_model_recall.py``` with different attention temperatures, it will concatenate new generations onto existing ones using the same parameters

    See a demo of model-generated recalls in ```recall_generation_demo.ipynb```. (Requires a GPU cluster. Tested on a compute node with three 40G A100 GPUs. Wall time of generating 30 recalls was 9 mins.) To run the demo, follow [these instructions](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to access the Llama3-8b-instruct model. Then follow the insturctions [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to setup commandline access to your huggingface account. 
2. Calculate how much recall explains about the story for model-generated recalls: (requires a cluster, output provided) ```bash model_recall_inference.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" 1```. Remember to change the stories you want to run inference on. If you generated new recalls using the same params, this inference code will append the new inference results onto existing ones. 
3. Calculate how much recall explains about the story for human recalls, and attention entropy from recall to story tokens: ```bash story_recall_inference.sh```(requires a cluster, output provided in ```generated/llama3-8b-instruct/prolific_data```)
3. The rate-distortion calculation is in ```rate distortion by attention scale-no annotations.ipynb```. This nb saves dictionaries for plotting in ```generated/llama3-8b-instruct/rate_distortion```. Rate distortion plots for all stories are in ```plot rate distortion_all stories together.ipynb```. 
    

## LLM temperature calibration (Supplement)
1. ```calibration.ipynb``` plots reliability diagrams (accuracy vs. confidence of next-token predictions) for Llama3-8b-instruct on story transcripts with various levels of temperature scaling. Computes expected calibration error on a held-out set of 20 stories from Moth Radio Hour, and check how those temperatures apply on our dataset of 8 stories.
2. To rerun analyses with a calibrated temperature, pass ```calibration_temp``` (e.g. 1.2) to ```run_story_even_split_analysis.sh``` (7th argument) and ```run_analyze_uniform_encoding.sh``` (7th argument). Outputs are saved with the suffix ```_caltemp_{temp}```.

## KL checks (Supplement)
1. ```KL vs. surprisal.ipynb``` compares token-level relative entropy KL(t||t-1) to surprisal (Supplemental Discussion). 
2. ```test_ce_kl_results.ipynb``` checks the assumption underlying the KL estimate between the generating and evaluating models (Fig. S25). 

## Verbatim recall simulation 
1. Generate stimuli and analysis in ```verbatim recall simulation.ipynb```
2. run ```verbatim_recall_inference.sh``` (Requires a GPU cluster, output provided)

## Determine attention head property (induction heads)
(Requires a GPU cluster. Output data is provided to demo subsequent steps.) Requires a GPU cluster: Use ```attention_try.ipynb``` to generate repeating stimuli and run inference to measure induction head score and duplicate token head score. Results are saved in ```generated/{model}/attention_head_test```. Dependency: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) 1.15.0: ```pip install transformer-lens==1.15.0```

