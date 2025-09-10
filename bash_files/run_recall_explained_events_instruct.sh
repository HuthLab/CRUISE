#!/bin/bash
# to run everything on TACC
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment")
fi
echo $stories
model=${1:-"Llama3-8b-instruct"} 

# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    python ../generate_recall_explained_events_tokens_instruct.py --story "$story" --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
    python ../get_recall_explained_events_ce_instruct.py --story "$story" --event_recall_concat --recall_event_concat --model "$model"
    python ../parse_recall_explained_events_ce_instruct.py --story "$story" --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
done

