#!/bin/bash
# run pairwise event inference for pairwise events
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array # if not passing stories via another bash file, pass multiple stories by: ""pieman" "alternateithicatom" "odetostepfather"", or single story like "sherlock"
else
    # Default array if no argument was provided
    stories=("adventuresinsayingyes" "inamoment" "legacy" "souls" "wheretheressmoke")
fi
echo $stories
model=${1:-"mistral-7b-instruct"} #"Llama3-8b-instruct"
echo $model

adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False
factor=${4:-None}
echo "factor = $factor"
split_by_token=${5:-None}
echo "split evenly by token = $split_by_token"

for story in "${stories[@]}"
do
    echo "Processing $story"
    if [ "$split_by_token" == "true" ]; then # split by tokens
        if [[ "$story" == "sherlock" ]]; then
            if [ "$factor" == "None" ]; then
                python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --twosessions --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_pairwise_event_ce.py --story "$story" --model "$model" --twosessions --split_story_by_tokens --save_dir ../generated 
            else
                python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --twosessions --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model" --twosessions --split_story_by_tokens --save_dir ../generated 
            fi
        else
            if [ "$adjusted" == "true" ]; then
                if [ "$factor" == "None" ]; then
                    python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --adjusted --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --story "$story" --model "$model"  --adjusted --split_story_by_tokens --save_dir ../generated 
                else
                    python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --adjusted --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model"  --adjusted --split_story_by_tokens --save_dir ../generated 
                fi
            else
                if [ "$factor" == "None" ]; then
                    python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --story "$story" --model "$model" --split_story_by_tokens --save_dir ../generated 
                else
                    python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --split_story_by_tokens --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model" --split_story_by_tokens --save_dir ../generated 
                fi
            fi
        fi
    else # split by duration
        if [[ "$story" == "sherlock" ]]; then
            if [ "$factor" == "None" ]; then
                python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --twosessions --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_pairwise_event_ce.py --story "$story" --model "$model" --twosessions --split_story_by_duration --save_dir ../generated 
            else
                python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --twosessions --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model" --twosessions --split_story_by_duration --save_dir ../generated 
            fi
        else
            if [ "$adjusted" == "true" ]; then
                if [ "$factor" == "None" ]; then
                    python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --adjusted --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --story "$story" --model "$model"  --adjusted --split_story_by_duration --save_dir ../generated 
                else
                    python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --adjusted --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model"  --adjusted --split_story_by_duration --save_dir ../generated 
                fi
            else
                if [ "$factor" == "None" ]; then
                    python ../generate_pairwise_event_stimuli.py --model "$model" --story "$story" --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --story "$story" --model "$model" --split_story_by_duration --save_dir ../generated 
                else
                    python ../generate_pairwise_event_stimuli.py --factor $factor --model "$model" --story "$story" --split_story_by_duration --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_pairwise_event_ce.py --factor $factor --story "$story" --model "$model" --split_story_by_duration --save_dir ../generated 
                fi
            fi
        fi
    fi
done
