#!/bin/bash
# all analyses to split the story into chunks of equal duration
# to run everything on TACC
if [ -n "$3" ]; then
    IFS=' ' read -r -a stories <<< "$3"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment")
fi

stories_string="${stories[*]}"
echo $stories_string

model=${1:-"Llama3-8b-instruct"} 
echo $model
random_recalls=${2:-false} 


# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    if [ "$random_recalls" == "true" ]; then
        python ../analyze_uniform_encoding.py --model "$model" --adjusted --random_recalls --story "$story" --save_dir ../../generated --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --exclusion_dir ../../behavior_data/exclusion --timing_dir ../../behavior_data/transcripts/timing
    else
        python ../analyze_uniform_encoding.py --model "$model" --adjusted --story "$story" --save_dir ../../generated --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --exclusion_dir ../../behavior_data/exclusion --timing_dir ../../behavior_data/transcripts/timing
    fi
done


