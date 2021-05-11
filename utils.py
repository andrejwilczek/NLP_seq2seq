import torch
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset
import numpy as np
import collections
import math


# Default word tokens
PAD_token = 0  # Pad token
# SOS_token = 1  # Start-of-sentence token
FURHAT_start = 1  # Start-of-utternace token Furhat
FURHAT_stop = 2  # End-of-utternace token Furhat
HUMAN_start = 3  # Start-of-utternace token human
HUMAN_stop = 4  # End-of-utternace token human
EOS_token = 5  # End-of-sentence token
UNK_token = 6  # Out of vocabulary token


def loadDataset(path):
    # Load the training data file
    df = pd.read_csv(path)

    # Create paris of input and target sequences
    X = df['input']
    Y = df['target']
    data_samples = list()
    X_train = list()
    Y_train = list()
    for i in range(len(X)):
        if isinstance(X[i], float):
            X[i] = '.'
        if isinstance(Y[i], float):
            Y[i] = '.'
        X[i] = remove_all_extra_spaces(X[i])
        Y[i] = remove_all_extra_spaces(Y[i])
        X_train.append(X[i])
        Y_train.append(Y[i])
        data_samples.append([X[i], Y[i]])

    return data_samples


def plotLoss(train_loss, valid_loss, n_iterations, save=False):
    epoch_vec = [epoch for epoch in range(n_iterations)]
    plt.plot(epoch_vec, train_loss, label='Training loss')
    plt.plot(epoch_vec, valid_loss, label='Validation loss')
    plt.legend()
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.title('')
    if save:
        plt.savefig('Training_loss.png')
    plt.show()


def remove_all_extra_spaces(string):
    return " ".join(string.split())


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index else UNK_token for word in sentence.split(' ')]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    matrix = []
    for i, seq in enumerate(l):
        matrix.append([])
        for token in seq:
            if token == PAD_token:
                matrix[i].append(0)
            else:
                matrix[i].append(1)
    return matrix


def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    # max_target_len = max([len(indexes) for indexes in indexes_batch])
    max_target_len = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    max_target_len = torch.tensor(max_target_len)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    # print(pair_batch)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        # print(pair)
        input_batch.append(pair[0])
        # print(input_batch)
        output_batch.append(pair[1])
        # print(output_batch)
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def teacher_decay(epoch, n_epochs, end_factor=0.7, e_min=0, e_max=1):
    slope = 1/(n_epochs*end_factor)
    epsilon = max(e_min, e_max-slope*epoch)
    return epsilon


def tsne_plot(voc, embedding, embedding_size, save=True):

    n_neighbors = 3
    most_common = 200
    counts = dict(sorted(voc.word2count.items(),
                         key=lambda item: item[1],
                         reverse=True))

    most_common_words = [word for word in list(counts)[:most_common]]
    snake_index = most_common_words.index('snake')
    snakes_index = most_common_words.index('snakes')
    ladder_index = most_common_words.index('ladder')
    ladders_index = most_common_words.index('ladders')
    yeah_index = most_common_words.index('yeah')
    game_index = most_common_words.index('game')
    luck_index = most_common_words.index('luck')

    parent_list = [snake_index, snakes_index, ladder_index, ladders_index,
                   yeah_index, game_index, luck_index]

    # reduce embeddings to 2d using tsne
    embeddings = np.empty((most_common, embedding_size))
    for i in range(most_common):
        embeddings[i, :] = torch.tensor(voc.word2index[most_common_words[i]])
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
    embeddings = tsne.fit_transform(embeddings)

    for parent in parent_list:

        embeddings_wo_parent = np.delete(embeddings, parent, 0)
        most_common_words_wo_parent = most_common_words.copy()
        most_common_words_wo_parent.pop(parent)

        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(embeddings_wo_parent)
        closest = (neigh.kneighbors([embeddings[parent]]))

        closest_dist = closest[0][0]
        closest_index = closest[1][0]
        neighbors = [most_common_words_wo_parent[index]
                     for index in closest_index]

        print("The closest words to {} is {} with distance {} \n".format(
            most_common_words[parent], neighbors, closest_dist))

    # plot embeddings
    fig, ax = plt.subplots(figsize=(30, 30))
    for i in range(most_common):
        ax.scatter(embeddings[i, 0], embeddings[i, 1])
        ax.annotate(most_common_words[i],
                    (embeddings[i, 0], embeddings[i, 1]))
    if save:
        plt.savefig('TSNE.png')
    plt.show()


def remove_specific_samples(text_set, idx_list, game_set=None, delta_set=None, history=True):

    if history:
        n_removed = 4
    else:
        n_removed = 2

    text_spec = list()
    game_spec = list()
    delta_spec = list()

    for i, idx in enumerate(idx_list):
        idx = idx-i*n_removed
        text_spec.append(text_set.pop(idx))

        if history:
            if game_set is not None:
                game_spec.append(game_set[idx])
                game_set = np.delete(game_set, [idx, idx+1, idx+2, idx+3], 0)
            if delta_set is not None:
                delta_spec.append(delta_set[idx])
                delta_set = np.delete(delta_set, [idx, idx+1, idx+2, idx+3], 0)
            for _ in range(3):
                text_set.pop(idx)

        else:
            text_set.pop(idx)
            if game_set is not None:
                game_spec.append(game_set[idx])
                game_set = np.delete(game_set, [idx, idx+1], 0)
            if delta_set is not None:
                delta_spec.append(delta_set[idx])
                delta_set = np.delete(delta_set, [idx, idx+1], 0)

    return text_set, game_set, delta_set, text_spec, game_spec, delta_spec


def create_tensor_dataset(voc, text_data, game_data, delta_data):
    text_data = batch2TrainData(voc, text_data)
    text_input, input_lengths, targets, mask, target_lengths = text_data

    # Convert to tensors with same dim(0)
    # print(game_data)
    # print(delta_data)
    text_input = text_input.permute(1, 0)
    targets = targets.permute(1, 0)
    mask = mask.permute(1, 0)
    # print(targets.shape)
    # print(mask.shape)
    # print(target_lengths.shape)
    # print(game_input.shape)
    if game_data is not None and delta_data is not None:
        game_input = torch.tensor(game_data, dtype=torch.float)
        delta_input = torch.tensor(delta_data, dtype=torch.float)
        dataset = TensorDataset(
            text_input, input_lengths, targets, mask, target_lengths, game_input, delta_input)
        return dataset
    elif game_data is not None:
        game_input = torch.tensor(game_data, dtype=torch.float)
        dataset = TensorDataset(
            text_input, input_lengths, targets, mask, target_lengths, game_input)
        return dataset
    elif delta_data is not None:
        delta_input = torch.tensor(delta_data, dtype=torch.float)
        dataset = TensorDataset(
            text_input, input_lengths, targets, mask, target_lengths, delta_input)
        return dataset
    else:
        dataset = TensorDataset(
            text_input, input_lengths, targets, mask, target_lengths)
        return dataset


def reshape_text_input(input_variable, target_variable, mask, target_len):
    input_variable = input_variable.permute(1, 0)
    target_variable = target_variable.permute(1, 0)
    mask = mask.permute(1, 0)
    max_target_len = max(target_len)
    return input_variable, target_variable, mask, max_target_len


def format_input(voc, use_history, use_game_state, use_delta_time, game_only, max_len_game_seq, game_dim):

    DIALOGUE_DIR = "dialogue_datasets/"

    if use_game_state:
        df_all = pd.read_csv('gamestate_datasets/game_state.csv')
        game_state = df_all.to_numpy().reshape(-1, max_len_game_seq, game_dim)
    else:
        game_state = None

    if use_history:
        text_path = 'dialogue_history.csv'
        delta_path = 'delta_time_history'
    else:
        text_path = 'dialogue.csv'
        delta_path = 'delta_time.csv'

    if use_delta_time:
        df_all_deltas = pd.read_csv(DIALOGUE_DIR+delta_path)
        delta_time = df_all_deltas.to_numpy()
    else:
        delta_time = None

    dialogue = loadDataset(DIALOGUE_DIR+text_path)

    # Specific testcases
    snake_indexes = [130, 414, 903, 1075, 1605, 1800, 2138, 2159, 2288, 2679]
    ladder_indexes = [268, 508, 654, 931, 1596, 1839, 2147, 2255, 2737]
    win_indexes = [316, 840, 1308, 1461, 1932, 2340, 2809]

    index_list = snake_indexes + ladder_indexes + win_indexes
    sorting_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    sorted_lists = sorted(zip(index_list, sorting_list))
    index_list = [x for x, y in sorted_lists]
    sorter = [y for x, y in sorted_lists]

    # Remove specific testcases from datset
    dialogue, game_state, delta_time, text_spec, game_spec, delta_spec = remove_specific_samples(
        dialogue, index_list, game_state, delta_time, use_history)

    # Create tensor dataset based on inputs
    if use_delta_time and use_game_state:
        sorted_input = sorted(zip(sorter, text_spec, game_spec, delta_spec))

        text_snake = [text for sort, text, game,
                      delta in sorted_input if sort == 0]
        game_snake = [game for sort, text, game,
                      delta in sorted_input if sort == 0]
        delta_snake = [delta for sort, text, game,
                       delta in sorted_input if sort == 0]

        text_ladder = [text for sort, text, game,
                       delta in sorted_input if sort == 1]
        game_ladder = [game for sort, text, game,
                       delta in sorted_input if sort == 1]
        delta_ladder = [delta for sort, text, game,
                        delta in sorted_input if sort == 1]

        text_win = [text for sort, text, game,
                    delta in sorted_input if sort == 2]
        game_win = [game for sort, text, game,
                    delta in sorted_input if sort == 2]
        delta_win = [delta for sort, text, game,
                     delta in sorted_input if sort == 2]

        test_snakes = create_tensor_dataset(
            voc, text_snake, game_snake, delta_snake)
        test_ladders = create_tensor_dataset(
            voc, text_ladder, game_ladder, delta_ladder)
        test_win = create_tensor_dataset(
            voc, text_win, game_win, delta_win)

        complete_dataset = create_tensor_dataset(
            voc, dialogue, game_state, delta_time)
    elif use_game_state:
        sorted_input = sorted(zip(sorter, text_spec, game_spec))

        text_snake = [text for sort, text, game in sorted_input if sort == 0]
        game_snake = [game for sort, text, game in sorted_input if sort == 0]

        text_ladder = [text for sort, text, game in sorted_input if sort == 1]
        game_ladder = [game for sort, text, game in sorted_input if sort == 1]

        text_win = [text for sort, text, game in sorted_input if sort == 2]
        game_win = [game for sort, text, game in sorted_input if sort == 2]

        test_snakes = create_tensor_dataset(
            voc, text_snake, game_snake, delta_time)
        test_ladders = create_tensor_dataset(
            voc, text_ladder, game_ladder, delta_time)
        test_win = create_tensor_dataset(
            voc, text_win, game_win, delta_time)

        complete_dataset = create_tensor_dataset(
            voc, dialogue, game_state, delta_time)

    elif use_delta_time:
        sorted_input = sorted(zip(sorter, text_spec, delta_spec))

        text_snake = [text for sort, text, delta in sorted_input if sort == 0]
        delta_snake = [delta for sort, text,
                       delta in sorted_input if sort == 0]

        text_ladder = [text for sort, text, delta in sorted_input if sort == 1]
        delta_ladder = [delta for sort, text,
                        delta in sorted_input if sort == 1]

        text_win = [text for sort, text, delta in sorted_input if sort == 2]
        delta_win = [delta for sort, text, delta in sorted_input if sort == 2]

        test_snakes = create_tensor_dataset(
            voc, text_snake, delta_snake)
        test_ladders = create_tensor_dataset(
            voc, text_ladder, delta_ladder)
        test_win = create_tensor_dataset(
            voc, text_win, delta_win)

        complete_dataset = create_tensor_dataset(
            voc, dialogue, delta_time)
    else:
        sorted_input = sorted(zip(sorter, text_spec))

        text_snake = [text for sort, text in sorted_input if sort == 0]
        text_ladder = [text for sort, text in sorted_input if sort == 1]
        text_win = [text for sort, text in sorted_input if sort == 2]

        test_snakes = create_tensor_dataset(
            voc, text_snake, game_state, delta_time)
        test_ladders = create_tensor_dataset(
            voc, text_ladder, game_state, delta_time)
        test_win = create_tensor_dataset(
            voc, text_win, game_state, delta_time)

        complete_dataset = create_tensor_dataset(
            voc, dialogue, game_state, delta_time)

    # split dataset inte train/valid/test
    train_size = int(0.8 * len(complete_dataset))
    valid_size = math.ceil(int(len(complete_dataset) - train_size)/2)
    test_size = math.floor(int(len(complete_dataset) - train_size)/2)
    assert(len(complete_dataset) == test_size+train_size+valid_size)

    # Shuffled datasets
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        complete_dataset, [train_size, valid_size, test_size])

    return train_dataset, validation_dataset, test_dataset, test_snakes, test_ladders, test_win
