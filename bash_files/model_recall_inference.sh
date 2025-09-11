#!/bin/bash
# run inference to get recall-story concat cross entropy and attention from story to recall
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather")
fi
echo $stories

model=${1:-"Llama3-8b-instruct"} 
echo $model
new_with_entropy="false" #"true"
prompt=${3:-1} # 1 is summary, 2 is as detailed as possible
for story in "${stories[@]}"
do
    echo "Processing $story"
    if [ "$new_with_entropy" == "true" ]; then
        python ../get_recall_tokens.py --model_recall_with_entropy --model "$model" --story "$story" --temp 0.7 --att_to_story_start --prompt_number $prompt --recall_only --recall_original_concat --original_recall_concat --save_dir ../generated --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated 
        python ../get_recall_logits.py --model_recall_with_entropy --model "$model" --file_name recall_tokens.pkl --story "$story" --entropy --temp 0.7 --att_to_story_start --prompt_number $prompt
        python ../get_recall_logits.py --model_recall_with_entropy --model "$model" --file_name original_recall_concat.pkl --story "$story" --entropy --attention --temp 0.7 --att_to_story_start --prompt_number $prompt
        python ../get_recall_logits.py --model_recall_with_entropy --model "$model" --file_name recall_original_concat.pkl --story "$story" --entropy --temp 0.7 --att_to_story_start --prompt_number $prompt
    else
        python ../get_recall_tokens.py --model "$model" --story "$story" --model_recall --temp 0.7 --att_to_story_start --prompt_number $prompt --recall_only --recall_original_concat --original_recall_concat --save_dir ../generated --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated 
        python ../get_recall_logits.py --model "$model" --file_name recall_tokens.pkl --story "$story" --entropy --model_recall --temp 0.7 --att_to_story_start --prompt_number $prompt
        # don't do attention inference for now to save time
        python ../get_recall_logits.py --model "$model" --file_name original_recall_concat.pkl --story "$story" --entropy --model_recall --temp 0.7 --att_to_story_start --prompt_number $prompt
        #python ../get_recall_logits.py --model "$model" --file_name original_recall_concat.pkl --story "$story" --entropy --attention --model_recall --temp 0.7 --att_to_story_start --prompt_number $prompt
        python ../get_recall_logits.py --model "$model" --file_name recall_original_concat.pkl --story "$story" --entropy --model_recall --temp 0.7 --att_to_story_start --prompt_number $prompt
    fi
done

