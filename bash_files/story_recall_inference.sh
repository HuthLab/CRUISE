#!/bin/bash
# run inference to get recall-story concat CE, and story-recall concat cross entropy and attention
stories=("pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment") 
model=${1:-"Llama3-8b-instruct"} 
echo $model
for story in "${stories[@]}"
do
    echo "Processing $story"
    python ../get_recall_tokens.py --model "$model" --story "$story" --recall_only --recall_original_concat --original_recall_concat --save_dir ../generated --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated 
    python ../get_recall_logits.py --model "$model" --file_name recall_tokens.pkl --story "$story" --entropy
    python ../get_recall_logits.py --model "$model" --file_name original_recall_concat.pkl --story "$story" --entropy --attention # also gets attention from recall to story
    python ../get_recall_logits.py --model "$model" --file_name recall_original_concat.pkl --story "$story" --entropy
done