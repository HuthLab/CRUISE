# CRUISE
This repository contains code used in the paper "Efficient uniform sampling explains non-uniform memory of narrative stories" by Jianing Mu, Alison R. Preston and Alexander G. Huth. 

### Software requirements
This repo has been tested on Linux Ubuntu 24.04.2 with Python 3.9.7 and 3.10.8
Dependencies: numpy, scipy, pandas, matplotlib, seaborn,pytorch (tested on 2.2.1 or 2.3.1), transformers (tested on 4.33.3 and 4.45.2). 
All code involving LLM inference was ran on a compute node with three 40G A100 GPUs. To demonstrate the analyses for users without GPU access, inference results are provided. 
See ```notebooks``` for demos and replications of figures in the main text. 

### Behavioral data parsing 
Raw Gorilla data to transcripts and segmentation, get metrics from recall coding
1. Generate participant_info spreadsheet by running ```python parse_output_spreadsheets.py```. Contains columns: prolific_id, gorilla_id, story, audio_task. See args for details, need to change the data directory for each new experiment. 
2. For tasks with multiple stories, run ```python group_audio_files.py``` to copy the audio files into their respective story directories. Saves under ```behavior_data/recall_audio/story```
3. Transcribe recall transcripts using ```transcribe_audio.py```
4. Extract segmentation, calculate segmentation consensus, comprehension accuracy with 
    ```python parse_behavioral_data.py```
    Use flag --exclude to exclude a pre-determined set of subjects. 
    For pie man, the ```Behavioral_data.ipynb``` computes the consensus segmentation, subjects' segmentation file and comprehension stats after exclusion
5. Run ```python parse_behavioral_data_combine.py``` (even if this story only has 1 experiment)
6. Exclusion criteria documented in ```Behavioral Results.ipynb``` and google spreadsheet Behavioral data masterlist. 
7. Run ```parse_behavioral_data_combine.py``` again with exclusion args '--exclude'
8. Run ```combine_recall_transcripts.py --story {story}``` to collect all checked recall transcripts into a csv file.
9. To get the event number of each detail in the story coding files, use the first few cells in the ```recall coding metrics.ipynb```. 
10. Extract clean recall transcripts using ```bash batch_check_transcripts.sh "story"```


### Split the story into chunks of equal durations, same number of chunks as num events (Fig.2 uniform encoding, and supplemental boundary analysis)
1. First run ```run_split_story_by_even_duration.sh``` locally to generate the unadjusted splits of the story. Output is under ```behavior_data/story_split_timing```. Then manually adjust for phrase boundaries. 
2. (Requires a GPU cluster. Output data is provided to demo subsequent steps.) Run ```run_story_even_split_analysis.sh```, packages inference code, runs both instruct and non-instruct to get I(Xi;R) (run_recall_explained_events). Also calculates H(X) (get_logits), I(Xi;Xj) (run_pairwise_events). Inference scripts called in this bash file uses --split_story_by_duration to indicate the even duration condition. 
3. Use ```run_analyze_uniform_encoding.sh``` to calculate CRUISE and other models. Generates dataframes for plotting
4. Use ```uniform encoding hypothesis combine stories-split story evenly by duration.ipynb``` to generate scatter plots 
5. Use ```uniform encoding hypothesis - split story evenly by duration - compare models.ipynb``` to generate bar plots of R^2
6. Use ```Uniform encoding hypothesis - by subject prevalence-split story evenly.ipynb``` to perform subject-level significance testing 


### Boundary analysis that splits the story into equal-duration or equal-token chunks (Fig.3 and supplemental results)
1. Split into equal token with 1.5xnumber of events 
    1. Generate windows ```split_story_by_tokens.py --story {story} --factor 1.5``` Outputs 'story_even_token_factor_%.1f.csv'%args.factor in behavior_data/story_split_timing
    2. Adjust for phrase boundaries manually, save them as 'story_even_token_factor_%.1f_adjusted.csv'%args.factor, send them back to TACC
    3. (Requires a GPU cluster. Output data is provided to demo subsequent steps.) Run ```bash run_story_even_split_analysis.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "true" "false" 1.5 "true"``` This calls run_split_story_by_even_duration.sh to align the adjusted chunks to the correct timing and tokens and recalculate whether each chunk is a boundary or not, then run the full LLM inference. Results are under pairwise_event/{story}/'story_split_tokens_factor_%.1f_adjusted'%args.factor. (default factor 1.5, meaning the number of windows = 1.5 times the number of events)
    4. Calculate CRUISE, surprisal weighted sampling and controls using ```uniform encoding hypothesis-split story evenly by tokens-split with factor.ipynb```
    5. Plot using ```split story by tokens - cleaned for plotting.ipynb```
2. Split into equal duration with 1.5 x number of events. 
    1. Generate chunks ```bash run_split_story_by_even_duration.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "false" 1.5 "false"```. Outputs 'story_even_duration_factor_%.1f.csv'%args.factor in behavior_data/story_split_timing
    2. Adjust for phrase boundaries manually, save them as 'story_even_duration_factor_%.1f_adjusted.csv'%args.factor
    3. Run ```bash run_story_even_split_analysis.sh "Llama3-8b-instruct" ""pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment"" "true" "false" 1.5 "false"``` This calls run_split_story_by_even_duration.sh to align the adjusted chunks to the correct timing and tokens, then run the full LLM inference. Results are under pairwise_event/{story}/'story_split_timing_factor_%.1f_adjusted'%args.factor
    4. Calculate CRUISE, surprisal weighted sampling and controls using code similar to the equal token split
    5. Plot using ```split story evenly by duration - chunks with boundary vs. no boundary cleaned for plotting.ipynb```

### Time courses of information properties around boundaries (Fig. 3jkl) and surprisal around boundaries vs. baseline (Supplement)
```CE around event boundaries vs. random chunks.ipynb```

### Mutual information of windows within event vs. across one event boundary (Fig. 4)
1. ```event_boundary_information.ipynb``` generates count balanced ablation stimuli in ```ablation/{model_name}/sliding_window_ablation/moth_stories```. 
2. (Requires a GPU cluster. Inference results are provided for subsequent analyses) Send stimuli to cluster for inference to obtain CE with ```sliding_ablation_entropy.py```. 
3. Analysis in ```event_boundary_information_cleaned.ipynb```

### Rate-distortion plot of LLM-generated recalls and human recalls (Fig. 5)
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
    

### Verbatim recall simulation 
1. Generate stimuli and analysis in ```verbatim recall simulation.ipynb```
2. run ```verbatim_recall_inference.sh``` (Requires a GPU cluster, output provided)

### Determine attention head property (induction heads)
(Requires a GPU cluster. Output data is provided to demo subsequent steps.) Requires a GPU cluster: Use ```attention_try.ipynb``` to generate repeating stimuli and run inference to measure induction head score and duplicate token head score. Results are saved in ```generated/{model}/attention_head_test```. Dependency: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) 1.15.0: ```pip install transformer-lens==1.15.0```

