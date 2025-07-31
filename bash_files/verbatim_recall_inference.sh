#!/bin/bash
# run inference to get recall-story concat and story-recall concat cross entropy for verbatim recalls
stories=("adventuresinsayingyes" "inamoment" "legacy" "souls" "wheretheressmoke") # ("adventuresinsayingyes" "inamoment" "legacy" "souls" "wheretheressmoke")
model=${1:-"Llama3-8b-instruct"} 
echo $model
for story in "${stories[@]}"
do
    echo "Processing $story"
    python ../get_recall_tokens.py --model "$model" --story "$story" --verbatim --recall_only --recall_original_concat --original_recall_concat --save_dir ../generated --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated 
    python ../get_recall_logits.py --model "$model" --file_name recall_tokens.pkl --story "$story" --entropy --verbatim
    python ../get_recall_logits.py --model "$model" --file_name original_recall_concat.pkl --story "$story" --entropy --verbatim
    python ../get_recall_logits.py --model "$model" --file_name recall_original_concat.pkl --story "$story" --entropy --verbatim
done