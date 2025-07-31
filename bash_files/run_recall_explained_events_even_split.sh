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

if [ "$split_by_token" == "true" ]; then # split by tokens
    # Loop through each story
    for story in "${stories[@]}"
    do
        echo "Processing $story"
        if [[ "$story" == "sherlock" ]]; then
            if [ "$factor" == "None" ]; then
                python ../generate_recall_explained_events_tokens.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_recall_explained_events_ce.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat  --twosessions --model "$model"
                python ../parse_recall_explained_events_ce.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat  --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
            else
                python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat  --twosessions --model "$model"
                python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat  --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
            fi
        else
            if [ "$adjusted" == "true" ]; then
                if [ "$random_recalls" == "true" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --adjusted --random_recalls --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --adjusted --random_recalls --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --adjusted --random_recalls --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                elif [ "$factor" != "None" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                else
                    python ../generate_recall_explained_events_tokens.py --story "$story" --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --adjusted --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                fi
            else
                if [ "$factor" != "None" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                else
                    python ../generate_recall_explained_events_tokens.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --split_story_by_tokens --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                fi
            fi
        fi
    done
else # split by duration
    # Loop through each story
    for story in "${stories[@]}"
    do
        echo "Processing $story"
        if [[ "$story" == "sherlock" ]]; then
            if [ "$factor" == "None" ]; then
                python ../generate_recall_explained_events_tokens.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_recall_explained_events_ce.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat  --twosessions --model "$model"
                python ../parse_recall_explained_events_ce.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat  --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
            else
                python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
                python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat  --twosessions --model "$model"
                python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat  --twosessions --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated --sherlock_transcript_dir ../sherlock
            fi
        else
            if [ "$adjusted" == "true" ]; then
                if [ "$random_recalls" == "true" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --adjusted --random_recalls --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --adjusted --random_recalls --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --adjusted --random_recalls --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                elif [ "$factor" != "None" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                else
                    python ../generate_recall_explained_events_tokens.py --story "$story" --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --adjusted --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                fi
            else
                if [ "$factor" != "None" ]; then
                    python ../generate_recall_explained_events_tokens.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --factor $factor --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                else
                    python ../generate_recall_explained_events_tokens.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                    python ../get_recall_explained_events_ce.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model"
                    python ../parse_recall_explained_events_ce.py --story "$story" --split_story_by_duration --event_recall_concat --recall_event_concat --model "$model" --save_dir ../generated --segmentation_dir ../behavior_data/segmentation --recall_transcript_dir ../behavior_data/recall_transcript --original_transcript_dir ../transcripts/moth_stories --moth_output_dir ../generated
                fi
            fi
        fi
    done
fi
