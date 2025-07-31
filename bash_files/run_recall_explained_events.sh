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
        python ../generate_recall_explained_events_tokens.py --story "$story" --event_recall_concat --recall_event_concat --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
        python ../get_recall_explained_events_ce.py --story "$story" --event_recall_concat --recall_event_concat  --twosessions --model "$model"
        python ../parse_recall_explained_events_ce.py --story "$story" --event_recall_concat --recall_event_concat  --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
    else
        python ../generate_recall_explained_events_tokens.py --story "$story" --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
        python ../get_recall_explained_events_ce.py --story "$story" --event_recall_concat --recall_event_concat --model "$model"
        python ../parse_recall_explained_events_ce.py --story "$story" --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
    fi
done

