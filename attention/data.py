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
    return {w:i for i,w in enumerate(open(data_path).read().splitlines())}, {i:w for i,w in enumerate(open(data_path).read().splitlines())}


def data_iterator(source_data_path, target_data_path, source_vocab, target_vocab, max_size, batch_size):
    with open(source_data_path, "rb") as f_in, open(target_data_path) as f_out:
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

            split_source = lsource.replace("\n", "").split(" ")
            split_target = ltarget.replace("\n", " </s>").split(" ")
            if len(split_source) <= max_size and len(split_target) + 1 <= max_size:

                source_text = [source_vocab[w] if w in source_vocab else source_vocab["<unk>"]
                               for w in split_source][::-1]
                target_text = [target_vocab["<s>"]] + [target_vocab[w] if w in target_vocab else target_vocab["<unk>"]
                               for w in split_target]
                source_data.append(pre_pad(source_text, source_vocab["<pad>"], max_size))
                target_data.append(post_pad(target_text, target_vocab["<pad>"], max_size))
                # ignore first word when computing length
                target_lengths.append(len(split_target))
                next_batch += 1

        if next_batch - prev_batch == batch_size:
            yield source_data, target_data, target_lengths
                