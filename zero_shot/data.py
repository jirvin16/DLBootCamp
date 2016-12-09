from __future__ import division
from __future__ import print_function


def pre_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[(max_len - len(lst)):] = lst
    return nlst


def post_pad(lst, pad_elt, max_len):
    nlst = [pad_elt]*max_len
    nlst[:len(lst)] = lst
    return nlst

def read_vocabulary(data_path):
	vocab_index = {}
	index_vocab = {}
	i = 0
	with open(data_path) as infile:
		for line in infile:
			for word in line.split():
				if word not in vocab_index:
					vocab_index[word] = i
					index_vocab[i] = word
					i += 1
	tokens = ["<pad>", "<s>", "</s>", "<unk>"]
	for j in range(len(tokens)):
		vocab_index[tokens[j]] = i + j
		index_vocab[i + j] = tokens[j]

	return vocab_index, index_vocab

def data_iterator(source_data_path, target_data_path, vocab, max_size, batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path, 'rb') as f_out:
        prev_batch     = 0
        next_batch     = 0
        source_data    = []
        target_data    = []
        target_lengths = []
        for i, (lsource, ltarget) in enumerate(zip(f_in, f_out)):

            if next_batch - prev_batch == batch_size:
                prev_batch = next_batch
                yield source_data, target_data, target_lengths
                source_data = []
                target_data = []
                target_lengths = []

            split_source = lsource.replace("\n", "").split()
            split_target = ltarget.replace("\n", "").split()
            if len(split_source) + 1 <= max_size and len(split_target) + 2 <= max_size:

                source_text = [vocab[w] if w in vocab else vocab["<unk>"] for w in split_source][::-1] + [vocab["</s>"]]
                target_text = [vocab["<s>"]] + [vocab[w] if w in vocab else vocab["<unk>"] for w in split_target] + [vocab["</s>"]]
                source_data.append(pre_pad(source_text, vocab["<pad>"], max_size))
                target_data.append(post_pad(target_text, vocab["<pad>"], max_size))
                # ignore first word when computing length
                target_lengths.append(len(target_text) - 1)
                next_batch += 1

        if next_batch - prev_batch == batch_size:
            yield source_data, target_data, target_lengths
                