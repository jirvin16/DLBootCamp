#!/bin/sh
# # Add spaces around punctuation, remove *, remove extra spaces, add target tokens to source
# python preprocess.py
# # Shuffle french-eng files together
# paste -d '*' /deep/group/dlbootcamp/jirvin16/fr-en/data.fr /deep/group/dlbootcamp/jirvin16/fr-en/data.en | shuf | awk -v FS="*" '{ print $1 > "/deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.fr" ; print $2 > "/deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.en" }'
# # Shuffle eng-german files together
# paste -d '*' /deep/group/dlbootcamp/jirvin16/en-de/data.en /deep/group/dlbootcamp/jirvin16/en-de/data.de | shuf | awk -v FS="*" '{ print $1 > "/deep/group/dlbootcamp/jirvin16/en-de/shuffled_data.en" ; print $2 > "/deep/group/dlbootcamp/jirvin16/en-de/shuffled_data.de" }'
# Undersample french-eng files
NUM_EXAMPLES=110000
TRAIN_SIZE=100000
VALID_SIZE=10000
head -n $NUM_EXAMPLES /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.fr > /deep/group/dlbootcamp/jirvin16/fr-en/sampled_data.fr
head -n $NUM_EXAMPLES /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.en > /deep/group/dlbootcamp/jirvin16/fr-en/sampled_data.en
# Undersample eng-german files
head -n $NUM_EXAMPLES /deep/group/dlbootcamp/jirvin16/en-de/shuffled_data.en > /deep/group/dlbootcamp/jirvin16/en-de/sampled_data.en
head -n $NUM_EXAMPLES /deep/group/dlbootcamp/jirvin16/en-de/shuffled_data.de > /deep/group/dlbootcamp/jirvin16/en-de/sampled_data.de
# Concatenate source
cat /deep/group/dlbootcamp/jirvin16/fr-en/sampled_data.fr /deep/group/dlbootcamp/jirvin16/en-de/sampled_data.en > /deep/group/dlbootcamp/jirvin16/data/data.fr_en
# Concatenate target
cat /deep/group/dlbootcamp/jirvin16/fr-en/sampled_data.en /deep/group/dlbootcamp/jirvin16/en-de/sampled_data.de > /deep/group/dlbootcamp/jirvin16/data/data.en_de
# Shuffle source and target together
paste -d '*' /deep/group/dlbootcamp/jirvin16/data/data.fr_en /deep/group/dlbootcamp/jirvin16/data/data.en_de | shuf | awk -v FS="*" '{ print $1 > "/deep/group/dlbootcamp/jirvin16/data/shuffled_data.fr_en" ; print $2 > "/deep/group/dlbootcamp/jirvin16/data/shuffled_data.en_de" }'
# Split into train and validation
# Source
head -n $TRAIN_SIZE /deep/group/dlbootcamp/jirvin16/data/shuffled_data.fr_en > /deep/group/dlbootcamp/jirvin16/data/train.fr_en
tail -n $VALID_SIZE /deep/group/dlbootcamp/jirvin16/data/shuffled_data.fr_en > /deep/group/dlbootcamp/jirvin16/data/valid.fr_en
# Target
head -n $TRAIN_SIZE /deep/group/dlbootcamp/jirvin16/data/shuffled_data.en_de > /deep/group/dlbootcamp/jirvin16/data/train.en_de
tail -n $VALID_SIZE /deep/group/dlbootcamp/jirvin16/data/shuffled_data.en_de > /deep/group/dlbootcamp/jirvin16/data/valid.en_de
# Combine files for word piece model
cat /deep/group/dlbootcamp/jirvin16/data/shuffled_data.fr_en /deep/group/dlbootcamp/jirvin16/data/shuffled_data.en_de > /deep/group/dlbootcamp/jirvin16/data/all
# Run wordpiece model
sh wordpiece.sh

# # # For french - english small pipeline
# # mkdir /deep/group/dlbootcamp/jirvin16/unilingual/
# head -n 55000 /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.fr > /deep/group/dlbootcamp/jirvin16/unilingual/temp.fr
# head -n 50000 /deep/group/dlbootcamp/jirvin16/unilingual/temp.fr > /deep/group/dlbootcamp/jirvin16/unilingual/train.fr
# tail -n 5000 /deep/group/dlbootcamp/jirvin16/unilingual/temp.fr > /deep/group/dlbootcamp/jirvin16/unilingual/valid.fr
# tail -n 2500 /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.fr > /deep/group/dlbootcamp/jirvin16/unilingual/test.fr
# head -n 55000 /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.en > /deep/group/dlbootcamp/jirvin16/unilingual/temp.en
# head -n 50000 /deep/group/dlbootcamp/jirvin16/unilingual/temp.en > /deep/group/dlbootcamp/jirvin16/unilingual/train.en
# tail -n 5000 /deep/group/dlbootcamp/jirvin16/unilingual/temp.en > /deep/group/dlbootcamp/jirvin16/unilingual/valid.en
# tail -n 2500 /deep/group/dlbootcamp/jirvin16/fr-en/shuffled_data.en > /deep/group/dlbootcamp/jirvin16/unilingual/test.en
# cat /deep/group/dlbootcamp/jirvin16/unilingual/temp.fr /deep/group/dlbootcamp/jirvin16/unilingual/test.fr /deep/group/dlbootcamp/jirvin16/unilingual/temp.en /deep/group/dlbootcamp/jirvin16/unilingual/test.en > /deep/group/dlbootcamp/jirvin16/unilingual/all