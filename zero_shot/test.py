actual_word = ['e', 'x', 'p', 'l', 'i', 'c', 'i', 't', 'l', 'y']
joined_word = "".join(actual_word)
best_merges = ["li", "ic", "ly", "it", "xp", "ex"]
while True:
	merge_found = False
	for merge in best_merges:
		if merge in joined_word:
			# replace tokens in string with \t
			index = joined_word.index(merge)
			joined_word = joined_word[:index] + "\t" * len(merge) + joined_word[index+len(merge):]
			# replace first token with merge, other tokens with \t
			actual_word[index] = merge
			actual_word[index+1:index+len(merge)] = "\t" * (len(merge) - 1)
			merge_found = True
			break
	# if no merges found, we are done
	if not merge_found:
		actual_word = filter(lambda a: a != "\t", actual_word)
		break