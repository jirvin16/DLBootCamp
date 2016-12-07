from __future__ import division
from __future__ import print_function

import string
import re
import os

from unicodedata import category
from unidecode import unidecode

def preprocess(infile_name, outfile_name, target_language=None):
	i = 0
	lines = []
	with open(infile_name) as infile:
		for line in infile:
			line = ''.join(unidecode(ch) if category(ch)[0] == 'P' else ch for ch in line.decode('utf8'))
			for ch in string.punctuation:
				line = line.replace(ch, " " + ch + " ")
			line = re.sub("[\t *]+", " ", line)
			i += 1
			if i % 10000 == 0:
				print(i)
			if target_language:
				lines.append("<{}> ".format(target_language) + line.strip() + "\n")
			else:
				lines.append(line.strip() + "\n")

	with open(outfile_name, 'wb') as outfile:
		outfile.write("".join([line.encode('utf8') for line in lines]))

if not os.path.isdir("/deep/group/dlbootcamp/jirvin16/data"):
	os.makedirs("/deep/group/dlbootcamp/jirvin16/data")
if not os.path.isdir("/deep/group/dlbootcamp/jirvin16/final_data"):
	os.makedirs("/deep/group/dlbootcamp/jirvin16/final_data")
file_triplets = [("/deep/group/dlbootcamp/jirvin16/fr-en/europarl-v7.fr-en.fr", "/deep/group/dlbootcamp/jirvin16/fr-en/data.fr", "en"),
				 ("/deep/group/dlbootcamp/jirvin16/fr-en/europarl-v7.fr-en.en", "/deep/group/dlbootcamp/jirvin16/fr-en/data.en", None),
				 ("/deep/group/dlbootcamp/jirvin16/en-de/train.en", "/deep/group/dlbootcamp/jirvin16/en-de/data.en", "de"),
				 ("/deep/group/dlbootcamp/jirvin16/en-de/train.de", "/deep/group/dlbootcamp/jirvin16/en-de/data.de", None),
				 ("/deep/group/dlbootcamp/jirvin16/fr-de/valid.fr", "/deep/group/dlbootcamp/jirvin16/data/test.fr", "de"),
				 ("/deep/group/dlbootcamp/jirvin16/fr-de/valid.de", "/deep/group/dlbootcamp/jirvin16/data/test.de", None)]

for infile_name, outfile_name, target_language in file_triplets:
	preprocess(infile_name, outfile_name, target_language)

