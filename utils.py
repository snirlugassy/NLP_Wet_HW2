from collections import defaultdict


def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def get_vocabs(list_of_paths):
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line != '\n':
                    splitted = line.split('\t')
                    if len(splitted) > 0:
                        word, pos_tag = splitted[1], splitted[3]
                        word_dict[word] += 1
                        pos_dict[pos_tag] += 1

    return word_dict, pos_dict
