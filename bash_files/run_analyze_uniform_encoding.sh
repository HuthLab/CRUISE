#!/bin/bash
# all analyses to split the story into chunks of equal duration or events
if [ -n "$9" ]; then
    IFS=' ' read -r -a stories <<< "$9"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment")
fi

stories_string="${stories[*]}"
echo $stories_string

model=${1:-"Llama3-8b-instruct"}
echo $model
random_recalls=${2:-false}
event=${3:-false} # if true split story into events, if false split into even-duration windows (same number of windows as events)
align_offset=${4:-false} # align event offset to start of next event, instead of using offset of last word of an event
split_by_token=${5:-false} # split into windows of even number of tokens
factor=${6:-None} # split into factor * number of events windows
calibration_temp=${7:-None} # calibration temperature
sr_exp=${8:-false} # sr experiment, runs both shrink and non shrink versions
# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    # Start building the argument list
    args=(--model "$model" --story "$story" --save_dir ../../generated --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --event_timing_dir ../../behavior_data/event_timing)

    # Add flags conditionally
    [ "$event" == "true" ] && args+=(--event)
    [ "$event" != "true" ] && args+=(--adjusted) # splitting by everything but events uses this flag
    [ "$random_recalls" == "true" ] && args+=(--random_recalls)
    [ "$align_offset" == "true" ] && args+=(--align_offset)
    [ "$split_by_token" == "true" ] && args+=(--split_by_token)
    [ "$factor" != "None" ] && args+=(--factor "$factor")
    [ "$calibration_temp" != "None" ] && args+=(--calibration_temp "$calibration_temp")
    if [[ "$sr_exp" == "true" ]]; then
        args+=(--sr_exp --sr_save_dir ../../exp2/generated --exclusion_dir ../../exp2/behavior_data/exclusion)
        # Run sr original
        echo "Running: python ../analyze_uniform_encoding.py ${args[*]}"
        python ../analyze_uniform_encoding.py "${args[@]}"
        # Run sr shrink
        echo "Running: python ../analyze_uniform_encoding.py ${args[*]} --shrink"
        python ../analyze_uniform_encoding.py "${args[@]}" --shrink
    else
        args+=(--exclusion_dir ../../behavior_data/exclusion)
        echo "Running: python ../analyze_uniform_encoding.py ${args[*]}"
        python ../analyze_uniform_encoding.py "${args[@]}"
    fi
done
