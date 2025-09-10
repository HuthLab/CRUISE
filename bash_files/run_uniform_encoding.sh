#!/bin/bash
# to run everything on TACC
stories=("wheretheressmoke" "adventuresinsayingyes" "inamoment")
#stories=("souls" "legacy")
#stories=("pieman" "alternateithicatom" "odetostepfather")

stories_string="${stories[*]}"

model=${1:-"Llama3-8b-instruct"} 
echo $model
python ../get_logits.py --model "$model" --save_dir ../generated --original_transcripts_dir ../transcripts/moth_stories
bash run_recall_explained_events.sh "$model" "$stories_string"
bash run_recall_explained_events_instruct.sh "$model" "$stories_string"
bash run_pairwise_events.sh "$model" "$stories_string"
