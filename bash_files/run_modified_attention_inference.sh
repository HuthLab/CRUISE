#!/bin/bash
# Using modified attention, run inference to get story-recall concat cross entropy and attention (if we want) from recall to story 
stories=("odetostepfather")
#stories=("legacy" "alternateithicatom" "odetostepfather" "souls" "wheretheressmoke" "adventuresinsayingyes")
#stories=("pieman" "alternateithicatom" "odetostepfather" "adventuresinsayingyes" "inamoment" "legacy" "souls" "wheretheressmoke")
model=${1:-"Llama3-8b-instruct"} 
attention_scales=(0.00001 0.00005 0.00007 0.0001 0.0002 0.0003 0.0004 0.0005 0.001 0.01)
echo $model
for story in "${stories[@]}"
do
    echo "Processing $story"
    for scale in "${attention_scales[@]}"
    do
        echo "$scale"
        python ../get_recall_logits.py --model "$model" --file_name original_recall_concat.pkl --attention --entropy --story "$story" --attention_scale "$scale"
    done
done

