#!/bin/bash
# to run everything on TACC
stories=("wheretheressmoke" "adventuresinsayingyes" "inamoment")
#stories=("souls" "legacy")
#stories=("pieman" "alternateithicatom" "odetostepfather")
#stories=("sherlock")
stories_string="${stories[*]}"

model=${1:-"mistral-7b-instruct"} #"Llama3-8b-instruct"
echo $model
if printf "%s\n" "${stories[@]}" | grep -q -x "sherlock"; then
    python ../get_logits.py --sherlock --twosessions --model "$model" --save_dir ../generated --original_transcripts_dir ../transcripts/moth_stories
else
    python ../get_logits.py --model "$model" --save_dir ../generated --original_transcripts_dir ../transcripts/moth_stories
fi
bash run_recall_explained_events.sh "$model" "$stories_string"
bash run_recall_explained_events_instruct.sh "$model" "$stories_string"
bash run_pairwise_events.sh "$model" "$stories_string"
