python learn_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/all -o /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -s 10000
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/train.fr_en -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/train.fr_en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/valid.fr_en -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/valid.fr_en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/test.fr -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/test.fr
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/train.en_de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/train.en_de
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/valid.en_de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/valid.en_de
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/test.de -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/test.de

python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/data/all -c /deep/group/dlbootcamp/jirvin16/final_data/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_data/segmented_all

# For french-english small pipeline
mkdir /deep/group/dlbootcamp/jirvin16/final_sample/
python learn_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/all -o /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -s 5000
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/train.fr -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/train.fr
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/valid.fr -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/valid.fr
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/test.fr -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/test.fr
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/train.en -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/train.en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/valid.en -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/valid.en
python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/test.en -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/test.en

python apply_bpe.py -i /deep/group/dlbootcamp/jirvin16/sample/all -c /deep/group/dlbootcamp/jirvin16/final_sample/wordpieces -o /deep/group/dlbootcamp/jirvin16/final_sample/segmented_all