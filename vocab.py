import pandas as pd
import numpy as np
import torch

# Default word tokens
PAD_token = 0  # Pad token
# SOS_token = 1  # Start-of-sentence token
FURHAT_start = 1  # Start-of-utternace token Furhat
FURHAT_stop = 2  # End-of-utternace token Furhat
HUMAN_start = 3  # Start-of-utternace token human
HUMAN_stop = 4  # End-of-utternace token human
EOS_token = 5  # End-of-sentence token
UNK_token = 6  # Out of vocabulary token


class Vocab():
    def __init__(self, make_embeddings=False):
        self.word2index = {"PAD": PAD_token, 'F': FURHAT_start, '/F': FURHAT_stop,
                           'H': HUMAN_start, '/H': HUMAN_stop, "EOS": EOS_token, "UNK": UNK_token}
        self.word2count = {"PAD": 0, "F": 0, "/F": 0,
                           "H": 0, "/H": 0, "EOS": 0, "UNK": 0}
        self.index2word = {PAD_token: "PAD",
                           FURHAT_start: "F", FURHAT_stop: "/F", HUMAN_start: "H", HUMAN_stop: "/H", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_words = 7  # Count PAD, SOS, EOS and UNK
        self.processData()
        if make_embeddings:
            print('\n --- Creating GLoVe embeddings from file --- \n')
            try:
                self.glove_embeddings = self.loadEmbeddings()
                self.embedding_matrix = self.createEmbeddingMatrix(
                    dimension=self.glove_embeddings['yes'].shape[0])
                # Convert weights to tensor
                self.embedding_weights = torch.tensor(
                    self.embedding_matrix, dtype=torch.float32)
                # Save embedding weights to file
                torch.save(self.embedding_weights, 'embedding_weights.pt')
                print(' --- Embeddings saved --- \n')
            except Exception as e:
                print(e)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def loadText(self):
        # Load the complete data file
        path = 'dialogue_datasets/dialogue.csv'
        df = pd.read_csv(path)
        return df

    def remove_all_extra_spaces(self, string):
        return " ".join(string.split())

    def processData(self):
        data_frame = self.loadText()
        text_list = list()
        for index, item in data_frame.iterrows():
            if isinstance(item['target'], float):
                item['target'] = ''
            text = item['target'].replace(".", " .").replace(
                ",", " ,").replace("?", " ?").replace("!", " !")
            text = self.remove_all_extra_spaces(text)
            text_list.append(text)
        for sentence in text_list:
            self.addSentence(sentence)

    def loadEmbeddings(self):
        # Load GLoVe embeddings
        print(' --- Reading embeddings --- \n')
        glove = pd.read_csv('Embedding_GLoVe/glove.6B.200d.txt', sep=" ",
                            quoting=3, header=None, index_col=0)
        glove_embeddings = {key: val.values for key, val in glove.T.items()}
        return glove_embeddings

    def createEmbeddingMatrix(self, dimension):
        print(' --- Creating embedding matrix --- \n')
        embedding_matrix = np.zeros((self.n_words, dimension))
        for word, index in self.word2index.items():
            if word in self.glove_embeddings:
                embedding_matrix[index] = self.glove_embeddings[word]
        return embedding_matrix
