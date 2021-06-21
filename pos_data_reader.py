class PosDataReader:
    def __init__(self, file):
        self.file = file
        self.sentences = []
        self.__read_data__()

    def __read_data__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = []
            for line in f:
                if line != '\n':
                    splitted = line.split('\t')
                    cur_sentence.append((splitted[1], splitted[3]))
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)
