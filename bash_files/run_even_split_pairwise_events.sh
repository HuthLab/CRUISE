#!/bin/bash
# run pairwise event inference for pairwise events
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array # if not passing stories via another bash file, pass multiple stories by: ""pieman" "alternateithicatom" "odetostepfather"", or single story like "sherlock"
else
    # Default array if no argument was provided
    stories=("adventuresinsayingyes" "inamoment" "legacy" "souls" "wheretheressmoke")
fi
echo $stories
model=${1:-"mistral-7b-instruct"} #"Llama3-8b-instruct"
echo $model

adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False
factor=${4:-None}
echo "factor = $factor"
split_by_token=${5:-None}
echo "split evenly by token = $split_by_token"
calibration_temp=${6:-None}
echo "Calibration temp = $calibration_temp"

for story in "${stories[@]}"
do
    echo "Processing $story"
    generate_stimuli_args=(--story "$story" --model "$model" --save_dir ../../generated --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --moth_output_dir ../../generated)
    get_ce_args=(--story "$story" --model "$model" --save_dir ../../generated)
 
    arg_sets=(generate_stimuli_args get_ce_args)

    for arr in "${arg_sets[@]}"; do
        # Add flags conditionally
        [ "$adjusted" == "true" ] && eval "$arr+=(--adjusted)"
        if [[ "$split_by_token" == "true" ]]; then
            eval "$arr+=(--split_story_by_tokens)"
        else
            eval "$arr+=(--split_story_by_duration)"
        fi
        [ "$factor" != "None" ] && eval "$arr+=(--factor \"$factor\")"
        if [[ "$arr" == "get_ce_args" ]]; then
            [ "$calibration_temp" != "None" ] && eval "$arr+=(--calibration_temp \"$calibration_temp\")"
        fi
    done
    echo "Running: python ../generate_pairwise_event_stimuli.py ${generate_stimuli_args[*]}"
    echo "Running: python ../get_pairwise_event_ce.py ${get_ce_args[*]}"
    python ../generate_pairwise_event_stimuli.py ${generate_stimuli_args[*]}
    python ../get_pairwise_event_ce.py ${get_ce_args[*]}
done
