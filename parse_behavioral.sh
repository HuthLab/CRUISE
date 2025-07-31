#!/bin/bash
# souls
python parse_behavioral_data.py --dir /home/jianing/generation/behavior_data/data_exp_156139-v10 --branch_a_story1_comp task-9pnb --nbranches 1
# send audios to celine then run:
python transcribe_audio.py --story souls
python parse_behavioral_data_combine.py --exp1 data_exp_156139-v10 --exp2 data_exp_156140-v6 --story souls
# determine exclusion in behavioral results then run:
python parse_behavioral_data_combine.py --exp1 data_exp_156139-v10 --exp2 data_exp_156140-v6 --story souls --exclude

# wheretheressmoke & legacy
# on local
python parse_output_spreadsheets.py --dir '/Users/mujianing/Desktop/Huth Lab/behavior_data/data_exp_166306-v4'
# on celine
python group_audio_files.py --dir /home/jianing/generation/behavior_data/data_exp_166306-v4
python transcribe_audio.py --story wheretheressmoke
python transcribe_audio.py --story legacy
python parse_behavioral_data.py --dir /home/jianing/generation/behavior_data/data_exp_166306-v4
python parse_behavioral_data_combine.py --story wheretheressmoke --exp1 data_exp_166306-v4 --exp2 none
python parse_behavioral_data_combine.py --story wheretheressmoke --exp1 data_exp_166306-v4 --exp2 none --exclude
python parse_behavioral_data_combine.py --story legacy --exp1 data_exp_166306-v4 --exp2 none
python parse_behavioral_data_combine.py --story legacy --exp1 data_exp_166306-v4 --exp2 none --exclude

# adventuresinsayingyes & inamoment
# on local
python parse_output_spreadsheets.py --dir '/Users/mujianing/Desktop/Huth Lab/behavior_data/data_exp_166306-v5'
# on celine
python group_audio_files.py --dir /home/jianing/generation/behavior_data/data_exp_166306-v5
python transcribe_audio.py --story adventuresinsayingyes
python transcribe_audio.py --story inamoment
python parse_behavioral_data.py --dir /home/jianing/generation/behavior_data/data_exp_166306-v5
python parse_behavioral_data_combine.py --story adventuresinsayingyes --exp1 data_exp_166306-v5 --exp2 none
python parse_behavioral_data_combine.py --story adventuresinsayingyes --exp1 data_exp_166306-v5 --exp2 none --exclude
python parse_behavioral_data_combine.py --story inamoment --exp1 data_exp_166306-v5 --exp2 none
python parse_behavioral_data_combine.py --story inamoment --exp1 data_exp_166306-v5 --exp2 none --exclude
