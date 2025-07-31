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
# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    if [[ "$story" == "sherlock" ]]; then
        python ../generate_pairwise_event_stimuli.py --story "$story" --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
        python ../get_pairwise_event_ce.py --story "$story" --twosessions --model "$model" --save_dir ../generated
    else
        python ../generate_pairwise_event_stimuli.py --story "$story" --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
        python ../get_pairwise_event_ce.py --story "$story" --model "$model" --save_dir ../generated
    fi
done