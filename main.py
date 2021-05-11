import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader,  TensorDataset
from encoder_decoder import *
from utils import *
from vocab import Vocab
import warnings
import wandb
import random
import math

warnings.filterwarnings("ignore")
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# Init the vocab class, set make_embedings to True if you have no saved glove embeddings for the dataset
voc = Vocab(make_embeddings=False)

#################################
#! Configure data structure
#################################
use_dialogue_history = False
use_game_state = False
use_delta_time = False
game_only = False


#################################
#! Configure models
#################################
hidden_size = 50
encoder_n_layers = 1
decoder_n_layers = 1
dropout = 0
batch_size = 10
load_embeddings = False      # Pretrained embeddings from GLoVe
bi_directional = True       # Bi-directional encoder


#################################
#! Configure training/optimization
#################################
train_model = True
clip = 5.0
teacher_forcing_ratio = 0
learning_rate = 0.001
decoder_learning_ratio = 1.0
n_iteration = 1
use_teacher_decay = False

#################################
#! Configure Evaluation
#################################
evaluate_on_test = True
n_eval = 40            # Number of test samples to evaluate
evaluate_specific = True
plot_embeddings = False


#################################
# wandb logging
#################################
# Start a new run
# wandb.init(project='dialogue-generation', entity='andrejwilczek')

# Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = learning_rate
# config.n_iteration = n_iteration
# config.clip = clip
# config.teacher_forcing_ratio = teacher_forcing_ratio
# config.batch_size = batch_size
# config.dropout = dropout
# config.hidden_size = hidden_size
# config.glove = load_embeddings
# config.encoder_layers = encoder_n_layers
# config.decoder_layers = decoder_n_layers
# config.decoder_learning_ratio = decoder_learning_ratio
# config.embedding_weights_requires_grad = embedding.weight.requires_grad
# config.dialogue_history = use_dialogue_history
# config.game_state = use_game_state

max_len_game_seq = 599
game_dim = 8
# Failsafe if poorly configured data structure
if game_only:
    use_game_state = True
    use_dialogue_history = False
    use_delta_time = False


if load_embeddings:
    print('--- Loading embedding weights from GLoVe ---')
    embedding_weights = torch.load('embedding_weights.pt')
    embedding = nn.Embedding.from_pretrained(
        embedding_weights)
    hidden_size = embedding_weights.size(1)
    print('\t Dimension of embeddings: ', hidden_size)
    # embedding.weight.requires_grad = False
else:
    print('--- Initializing untrained embeddings ---')
    embedding = nn.Embedding(voc.n_words, hidden_size)


# Init encoder/decoder
print('--- Building encoder and decoder ---')
encoder = EncoderRNN(hidden_size, embedding,
                     encoder_n_layers, dropout, use_game_state, use_dialogue_history, use_delta_time, bi_directional, game_dim, game_only)
decoder = DecoderRNN(embedding, hidden_size, voc.n_words,
                     decoder_n_layers, dropout)

# Send encoder/decoder to device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('--- Models built ---')
print(encoder)
print(decoder)

# Create dataloaders
train_dataset, validation_dataset, test_dataset, test_snakes, test_ladders, test_win = format_input(voc, use_dialogue_history, use_game_state, use_delta_time,
                                                                                                    game_only, max_len_game_seq, game_dim)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(
    validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, drop_last=True)

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('--- Building optimizers ---')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(
    decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


#################################
# Training
#################################
if train_model:
    print("--- Starting Training --- \n")
    training_loss, validation_loss, avg_train_loss, avg_valid_loss = trainIters(voc, train_dataloader, valid_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                                                                encoder_n_layers, decoder_n_layers,  n_iteration, batch_size, clip, use_teacher_decay, teacher_forcing_ratio)
else:
    checkpoint = torch.load(
        'Text+Game no history/checkpoint epoch_10.zip', map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder_optimizer.load_state_dict(
        checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(
        checkpoint['decoder_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loading epoch {epoch} with loss {loss}')


if plot_embeddings:
    tsne_plot(voc, embedding, hidden_size)

#################################
# Evaluation
#################################
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Evaluate on test data
if evaluate_on_test:
    evaluateTestset(encoder, decoder, searcher, voc, test_dataloader, n_eval)

# Evaluate on specific test cases
if evaluate_specific:
    print('\n --- Evaluating snakes --- \n')
    specific_eval(encoder, decoder, searcher, voc, test_snakes)
    print('\n --- Evaluating ladders --- \n')
    specific_eval(encoder, decoder, searcher, voc, test_ladders)
    print('\n --- Evaluating wins --- \n')
    specific_eval(encoder, decoder, searcher, voc, test_win)
