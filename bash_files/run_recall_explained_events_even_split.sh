#!/bin/bash
# to run everything on TACC
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("pieman" "alternateithicatom" "odetostepfather")
fi
echo $stories

model=${1:-"mistral-7b-instruct"} #"Llama3-8b-instruct"
echo $model
adjusted=${3:-false} # whether to evalutate on the adjusted version. Defaults to False
echo "adjusted = $adjusted"
random_recalls=${4:-false} # whether to evaluate on the randomly selected recalls from other stories
echo "random_recalls = $random_recalls"
factor=${5:-None} # multiplication factor, nchunks = factor * nevents
echo "factor = $factor"
split_by_token=${6:-None}
echo "split evenly by token = $split_by_token"
calibration_temp=${7:-None}
echo "Calibration temp = $calibration_temp"
sr_exp=${8:-false} # speech rate exp, uses sr transcripts instead of the original
echo "speech rate exp = $sr_exp"

for story in "${stories[@]}"
do
    echo "Processing $story"
    tokens_args=(--story "$story" --model "$model" --event_recall_concat --recall_event_concat --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --moth_output_dir ../../generated)
    get_ce_args=(--story "$story" --model "$model" --event_recall_concat --recall_event_concat)
    parse_ce_args=(--story "$story" --model "$model"  --event_recall_concat --recall_event_concat --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --moth_output_dir ../../generated)

    arg_sets=(tokens_args get_ce_args parse_ce_args)

    for arr in "${arg_sets[@]}"; do
        # Add flags conditionally
        [ "$random_recalls" == "true" ] && eval "$arr+=(--random_recalls)"
        [ "$adjusted" == "true" ] && eval "$arr+=(--adjusted)"
        if [[ "$split_by_token" == "true" ]]; then
            eval "$arr+=(--split_story_by_tokens)"
        else
            eval "$arr+=(--split_story_by_duration)"
        fi
        [ "$factor" != "None" ] && eval "$arr+=(--factor \"$factor\")"
        if [[ "$arr" == "parse_ce_args" || "$arr" == "get_ce_args" ]]; then
            [ "$calibration_temp" != "None" ] && eval "$arr+=(--calibration_temp \"$calibration_temp\")"
        fi
    done

    if [[ "$sr_exp" == "true" ]]; then
        tokens_args+=(--sr_exp --sr_save_dir ../../exp2/generated --recall_transcript_dir ../../exp2/behavior_data/recall_transcript)
        parse_ce_args+=(--sr_exp --sr_save_dir ../../exp2/generated --recall_transcript_dir ../../exp2/behavior_data/recall_transcript)
        get_ce_args+=(--save_dir ../../exp2/generated)

    else
        tokens_args+=(--recall_transcript_dir ../../behavior_data/recall_transcript)
        parse_ce_args+=(--recall_transcript_dir ../../behavior_data/recall_transcript)
        get_ce_args+=(--save_dir ../../generated)
    fi


    # Optional debugging log
    echo "Running: python ../generate_recall_explained_events_tokens.py ${tokens_args[*]}"
    echo "Running: python ../get_recall_explained_events_ce.py ${get_ce_args[*]}"
    echo "Running: python ../parse_recall_explained_events_ce.py ${parse_ce_args[*]}"
    python ../generate_recall_explained_events_tokens.py ${tokens_args[*]}
    python ../get_recall_explained_events_ce.py ${get_ce_args[*]}
    python ../parse_recall_explained_events_ce.py ${parse_ce_args[*]}
done
