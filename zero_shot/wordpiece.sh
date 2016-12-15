#!/bin/sh
python learn_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/all -o /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -s 16000
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/train.fr_en -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/train.fr_en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/valid.fr_en -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/valid.fr_en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/test.fr -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/test.fr
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/train.en_de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/train.en_de
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/valid.en_de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/valid.en_de
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/test.de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/test.de

# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/all -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/segmented_all
cat /deep/group/dlbootcamp/jirvin16/final_data/train.fr_en /deep/group/dlbootcamp/jirvin16/final_data/valid.fr_en /deep/group/dlbootcamp/jirvin16/final_data/train.en_de /deep/group/dlbootcamp/jirvin16/final_data/valid.en_de > /deep/group/dlbootcamp/jirvin16/final_data/segmented_all

# # For french-english small pipeline
# mkdir /deep/group/dlbootcamp/jirvin16/final_unilingual/
# python learn_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/all -o /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -s 5000
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/train.fr -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/train.fr
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/valid.fr -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/valid.fr
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/test.fr -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/test.fr
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/train.en -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/train.en
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/valid.en -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/valid.en
# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/test.en -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/test.en

# python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/unilingual/all -c /deep/group/dlbootcamp/jirvin16/final_unilingual/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_unilingual/segmented_all