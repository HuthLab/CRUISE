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
    args=(--story "$story" --model "$model" --save_dir ../generated)
 
    # Add flags conditionally
    [ "$adjusted" == "true" ] && args+=(--adjusted)
    if [[ "$split_by_token" == "true" ]]; then
        args+=(--split_story_by_tokens)
    else
        args+=(--split_story_by_duration)
    fi
    [ "$factor" != "None" ] && args+=(--factor "$factor")

    echo "Running: python ../get_recombine_event_ce.py ${args[*]}"
    python ../get_recombine_event_ce.py ${args[*]}
done