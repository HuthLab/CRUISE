#!/bin/bash
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather")
fi
echo $stories

model=${1:-"Llama3-8b-instruct"} 
echo $model
adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False
factor=${4:-None}
echo "factor = $factor"
split_by_token=${5:-None}
echo "split evenly by token = $split_by_token"

for story in "${stories[@]}"
do
    echo "Processing $story"
    if [ "$split_by_token" == "true" ]; then # split by tokens
        if [ "$factor" == "None" ]; then
            if [ "$adjusted" == "true" ]; then
                python ../get_recombine_event_ce.py --model "$model" --story "$story" --adjusted --split_story_by_tokens
            else
                python ../get_recombine_event_ce.py --model "$model" --story "$story" --split_story_by_tokens
            fi
        else # factor provided
            if [ "$adjusted" == "true" ]; then
                python ../get_recombine_event_ce.py --factor $factor --model "$model" --story "$story" --adjusted --split_story_by_tokens
            else
                python ../get_recombine_event_ce.py --factor $factor --model "$model" --story "$story" --split_story_by_tokens
            fi
        fi
    else
        if [ "$factor" == "None" ]; then
            if [ "$adjusted" == "true" ]; then
                python ../get_recombine_event_ce.py --model "$model" --story "$story" --adjusted --split_story_by_duration
            else
                python ../get_recombine_event_ce.py --model "$model" --story "$story" --split_story_by_duration
            fi
        else # factor provided
            if [ "$adjusted" == "true" ]; then
                python ../get_recombine_event_ce.py --factor $factor --model "$model" --story "$story" --adjusted --split_story_by_duration
            else
                python ../get_recombine_event_ce.py --factor $factor --model "$model" --story "$story" --split_story_by_duration
            fi
        fi
    fi
done