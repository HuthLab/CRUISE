#!/bin/bash
# all analyses to split the story into chunks of equal duration or tokens
# to run everything on TACC
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment")
    #stories=("pieman" "alternateithicatom" "odetostepfather" "legacy" "souls")
fi

stories_string="${stories[*]}"
echo $stories_string

model=${1:-"Llama3-8b-instruct"} 
echo $model

adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False
echo "adjusted = $adjusted"

random_recalls=${4:-false} # whether to evaluate on the randomly selected recalls from other stories
echo "random recalls = $random_recalls"

factor=${5:-None} # float, determines how many chunks to split into. num chunks = factor * num events
echo "factor = $factor"

split_by_token=${6:-None}
echo "split evenly by token = $split_by_token"

if [ "$random_recalls" == "true" ]; then # cuz the mutual information between parts of the story has already been calculated
    bash run_recall_explained_events_even_split.sh "$model" "$stories_string" "$adjusted" "$random_recalls" $factor "$split_by_token"
    bash run_recall_explained_events_instruct_even_split.sh "$model" "$stories_string" "$adjusted" "$random_recalls" $factor "$split_by_token"
else
    bash run_split_story_by_even_duration.sh "$model" "$stories_string" "$adjusted" $factor "$split_by_token" # create the even split chunks
    bash run_even_split_pairwise_events.sh "$model" "$stories_string" "$adjusted" $factor "$split_by_token"
    bash run_even_split_event_ce.sh "$model" "$stories_string" "$adjusted" $factor "$split_by_token"
    bash run_recall_explained_events_even_split.sh "$model" "$stories_string" "$adjusted" "$random_recalls" $factor "$split_by_token"
    bash run_recall_explained_events_instruct_even_split.sh "$model" "$stories_string" "$adjusted" "$random_recalls" $factor "$split_by_token"
fi