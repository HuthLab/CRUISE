#!/bin/bash
# to run everything on TACC
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather")
fi
echo $stories

model=${1:-"mistral-7b-instruct"} #"Llama3-8b-instruct"
echo $model

adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False

factor=${4:-None}
echo "factor = $factor"

split_by_token=${5:-None}
echo "split evenly by token = $split_by_token"

# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    if [ "$split_by_token" == "true" ]; then # split by tokens
        if [ "$factor" == "None" ]; then
            python ../add_token_idx_to_story_even_duration_split.py --even_tokens --parse_adjusted --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
        else
            python ../add_token_idx_to_story_even_duration_split.py --even_tokens --parse_adjusted --factor $factor --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
        fi
    else # split by duration
        if [ "$adjusted" == "true" ]; then # adjusted
            if [ "$factor" == "None" ]; then
                # comment out cuz this only needs to be done once for all models
                python ../split_story_by_timing.py --story "$story" --parse_adjusted --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
                python ../add_token_idx_to_story_even_duration_split.py --parse_adjusted --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
            else
                python ../split_story_by_timing.py --story "$story" --parse_adjusted --factor $factor --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
                python ../add_token_idx_to_story_even_duration_split.py --parse_adjusted --factor $factor --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
            fi
        else
            if [ "$factor" == "None" ]; then
                # unadjusted
                python ../split_story_by_timing.py --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
                python ../add_token_idx_to_story_even_duration_split.py --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
            else
                python ../split_story_by_timing.py --story "$story" --factor $factor --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
                python ../add_token_idx_to_story_even_duration_split.py --factor $factor --model "$model" --story "$story" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --timing_dir ../transcripts/timing --event_timing_dir ../behavior_data/story_split_timing
            fi
        fi
    fi
done