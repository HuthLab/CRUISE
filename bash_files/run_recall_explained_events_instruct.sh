#!/bin/bash
# to run everything on TACC
if [ -n "$2" ]; then
    IFS=' ' read -r -a stories <<< "$2"  # Split argument back into an array
else
    # Default array if no argument was provided
    stories=("alternateithicatom" "odetostepfather" "legacy" "souls" "wheretheressmoke" "adventuresinsayingyes" "inamoment")
fi
echo $stories
model=${1:-"Llama3-8b-instruct"} 
calibration_temp=${3:-None}
echo "Calibration temp = $calibration_temp"
sr_exp=${4:-false} # speech rate exp, uses sr transcripts instead of the original
echo "speech rate exp = $sr_exp"

# Loop through each story
for story in "${stories[@]}"
do
    echo "Processing $story"
    tokens_args=(--story "$story" --model "$model" --event_recall_concat --recall_event_concat --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --moth_output_dir ../../generated)
    get_ce_args=(--story "$story" --model "$model" --event_recall_concat --recall_event_concat)
    parse_ce_args=(--story "$story" --model "$model"  --event_recall_concat --recall_event_concat --segmentation_dir ../../behavior_data/segmentation --original_transcript_dir ../../behavior_data/transcripts/moth_stories --moth_output_dir ../../generated)

    if [[ "$sr_exp" == "true" ]]; then
        tokens_args+=(--sr_exp --sr_save_dir ../../exp2/generated --recall_transcript_dir ../../exp2/behavior_data/recall_transcript)
        parse_ce_args+=(--sr_exp --sr_save_dir ../../exp2/generated --recall_transcript_dir ../../exp2/behavior_data/recall_transcript)
        get_ce_args+=(--save_dir ../../exp2/generated)

    else
        tokens_args+=(--recall_transcript_dir ../../behavior_data/recall_transcript)
        parse_ce_args+=(--recall_transcript_dir ../../behavior_data/recall_transcript)
        get_ce_args+=(--save_dir ../../generated)
    fi
    if [[ "$calibration_temp" != "None" ]]; then
        get_ce_args+=(--calibration_temp $calibration_temp)
        parse_ce_args+=(--calibration_temp $calibration_temp)
    fi
        
    echo "Running: python ../generate_recall_explained_events_tokens_instruct.py ${tokens_args[*]}"
    echo "Running: python ../get_recall_explained_events_ce_instruct.py ${get_ce_args[*]}"
    echo "Running: python ../parse_recall_explained_events_ce_instruct.py ${parse_ce_args[*]}"
    python ../generate_recall_explained_events_tokens_instruct.py ${tokens_args[*]}
    python ../get_recall_explained_events_ce_instruct.py ${get_ce_args[*]}
    python ../parse_recall_explained_events_ce_instruct.py ${parse_ce_args[*]}
done

