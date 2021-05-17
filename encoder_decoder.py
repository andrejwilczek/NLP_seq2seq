import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import batch2TrainData, indexesFromSentence, teacher_decay, reshape_text_input

import math
import random
import numpy as np

import wandb

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Default word tokens
PAD_token = 0  # Pad token
FURHAT_start = 1  # Start-of-utternace token Furhat
FURHAT_stop = 2  # End-of-utternace token Furhat
HUMAN_start = 3  # Start-of-utternace token human
HUMAN_stop = 4  # End-of-utternace token human
EOS_token = 5  # End-of-sentence token
UNK_token = 6  # Out of vocabulary token


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.1,  game_state=False, history=False, delta_time=False, bidirectional=False, game_dim=8, game_only=False):
        super(EncoderRNN, self).__init__()

        # Parameters
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_game_state = game_state
        self.game_dim = game_dim
        self.history = history
        self.use_delta_time = delta_time
        self.game_only = game_only

        # Layers
        self.embedding = embedding

        if self.bidirectional:
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        # Game only
        if self.game_only:
            self.game_gru = nn.GRU(
                self.game_dim, self.hidden_size, self.n_layers)

        # Dialogue game state and delta_time:
        elif self.use_game_state and self.use_delta_time:
            self.game_gru = nn.GRU(
                self.game_dim, self.hidden_size, self.n_layers)
            if self.history:
                self.fc = nn.Linear(self.hidden_size*2+3, self.hidden_size)
            else:
                self.fc = nn.Linear(self.hidden_size*2+1, self.hidden_size)

        # Dialogue and game state
        elif self.use_game_state:
            self.game_gru = nn.GRU(
                self.game_dim, self.hidden_size, self.n_layers)
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)

        # Dialogue and delta_time
        elif self.use_delta_time:
            if self.history:
                self.fc = nn.Linear(self.hidden_size+3, self.hidden_size)
            else:
                self.fc = nn.Linear(self.hidden_size+1, self.hidden_size)

        # Dialogue only
        else:
            self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_seq=None, input_lengths=None, game_seq=None, delta=None, game_hidden=None, hidden=None):

        if self.game_only:
            # Only use game state
            outputs, hidden = self.game_gru(game_seq, game_hidden)
        else:
            # Convert word indexes to embeddings
            embedded = self.embedding(input_seq)

            # Pack padded batch of sequences for RNN module
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), enforce_sorted=False)

            # Forward pass through GRU
            outputs, hidden = self.gru(packed, hidden)

            # Unpack padding
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

            # Sum bidirectional GRU outputs if bidirectional is True
            if self.bidirectional:
                outputs = outputs[:, :, :self.hidden_size] + \
                    outputs[:, :, self.hidden_size:]
                hidden = hidden[:self.n_layers, :, :] + \
                    hidden[self.n_layers:, :, :]

            # text, gamestate and  time information
            if self.use_game_state and self.use_delta_time:
                # Forward game data through game GRU
                game_out, game_hidden = self.game_gru(game_seq, game_hidden)
                # Concat text and game hidden states
                sum_hid = torch.cat((hidden, game_hidden), 2)
                # Add dim to delta and repeat tensor for number of layers
                delta = delta.unsqueeze(0)
                delta = delta.repeat(self.n_layers, 1, 1)
                # Concat delta to hidden state
                sum_hid = torch.cat((sum_hid, delta), 2)
                # Forward hidden through fully connected layer
                self.fc.float()
                hidden = self.fc(sum_hid)

            # text and gamestate
            elif self.use_game_state:
                # Forward game data through game GRU
                game_out, game_hidden = self.game_gru(game_seq, game_hidden)
                # Concat text and game hidden states
                sum_hid = torch.cat((hidden, game_hidden), 2)
                # Forward hidden through fully connected layer
                self.fc.float()
                hidden = self.fc(sum_hid)

            # text and time information
            elif self.use_delta_time:
                # Add dim to delta and repeat tensor for number of layers
                delta = delta.unsqueeze(0)
                delta = delta.repeat(self.n_layers, 1, 1)
                # Concat delta to hidden state
                sum_hid = torch.cat((hidden, delta), 2)
                # Forward hidden through fully connected layer
                self.fc.float()
                hidden = self.fc(sum_hid)

        # Return output and final hidden state
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Layers
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        # Get embedding of current input word
        embedded = self.embedding(input_step)

        # Forward through GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output = rnn_output.squeeze(0)

        # Forward thrrough fully connected layer
        output = self.fc(rnn_output)

        # Forward through softmax for prediction

        # use this with maskNLLLoss() function
        # output = F.softmax(output, dim=1)

        # use this with NLLLoss()
        output = F.log_softmax(output, dim=1)

        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, target_seq, game_seq=None, delta=None):

        # Forward input through encoder model
        if self.encoder.use_game_state and self.encoder.use_delta_time:
            encoder_outputs, encoder_hidden = self.encoder(
                input_seq, input_length, game_seq, delta)
        elif self.encoder.use_game_state:
            encoder_outputs, encoder_hidden = self.encoder(
                input_seq, input_length, game_seq)
        elif self.encoder.use_delta_time:
            encoder_outputs, encoder_hidden = self.encoder(
                input_seq, input_length, delta=delta)
        else:
            encoder_outputs, encoder_hidden = self.encoder(
                input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize decoder input with player token from target
        decoder_input = target_seq[0]
        decoder_input = decoder_input.unsqueeze(0)

        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        # Iteratively decode one word token at a time
        for _ in range(max_length):

            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # Return collections of word tokens and scores
        return all_tokens, all_scores


def maskNLLLoss(inp, target, mask):

    nTotal = mask.sum()
    NLL = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    mask = torch.tensor(mask, dtype=torch.bool)
    loss = NLL.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,  encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip,  teacher_forcing_ratio,  game_tensor=None, delta=None):
    """
    Performs forward and backward pass for one batch of training samples. 
    """

    # Ensure train mode
    encoder.train()
    decoder.train()

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    if delta is not None:
        delta = delta.to(device)
    if game_tensor is not None:
        game_tensor = game_tensor.to(device)

    # Initialize variables
    loss = 0
    loss_layer = nn.NLLLoss(ignore_index=0)
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    if encoder.use_game_state and encoder.use_delta_time:
        encoder_outputs, encoder_hidden = encoder(
            input_variable, lengths, game_tensor, delta)
    elif encoder.use_game_state:
        if encoder.game_only:
            encoder_outputs, encoder_hidden = encoder(game_seq=game_tensor)
        else:
            encoder_outputs, encoder_hidden = encoder(
                input_variable, lengths, game_tensor)
    elif encoder.use_delta_time:
        encoder_outputs, encoder_hidden = encoder(
            input_variable, lengths, delta=delta)
    else:
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Decoder input is the first token in target (Player start token)
    decoder_input = target_variable[0, :]
    decoder_input = decoder_input.unsqueeze(0)
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len-1):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss
            # NLLLoss()
            mask_loss = loss_layer(decoder_output, target_variable[t+1])
            nTotal = mask.sum().item()

            # maskNLLLoss()
            # mask_loss, nTotal = maskNLLLoss(
            #     decoder_output, target_variable[t], mask[t])

            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        else:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss
            # NLLLoss()
            # print(decoder_output)
            # print(target_variable[t+1])
            # print(target_variable[t])
            mask_loss = loss_layer(decoder_output, target_variable[t+1])
            nTotal = mask.sum().item()
            # maskNLLLoss()
            # mask_loss, nTotal = maskNLLLoss(
            #     decoder_output, target_variable[t], mask[t])

            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # quit()
    # Perform backpropatation
    loss.backward()

    # Clip gradients
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(voc, train_dataloader, valid_dataloader,  encoder, decoder, encoder_optimizer, decoder_optimizer,  encoder_n_layers, decoder_n_layers,  n_epochs, batch_size, clip, use_teacher_decay, teacher_forcing_ratio):

    # Initializations
    start_epoch = 1
    print_train_loss = 0
    print_valid_loss = 0
    plot_train_loss = list()
    plot_valid_loss = list()
    plot_avg_train_loss = list()
    plot_avg_valid_loss = list()
    valid_batch_size = batch_size

    # Training loop
    print("--- Training ---")
    for epoch in range(start_epoch, n_epochs + 1):

        # Determine if we are using teacher forcing this epoch
        if use_teacher_decay:
            teacher_forcing_ratio = teacher_decay(epoch, n_epochs+1)
            print('teacher forcing: ', teacher_forcing_ratio)

        for batch_number, train_batch in enumerate(train_dataloader):

            # Get next batch of validation data
            valid_batch = next(iter(valid_dataloader))

            if encoder.use_game_state and encoder.use_delta_time:

                # Training set
                input_variable, lengths, target_variable, mask, target_len, game_state, delta = train_batch
                game_state = game_state.permute(1, 0, 2)
                input_variable,  target_variable, mask, max_target_len = reshape_text_input(
                    input_variable,  target_variable, mask, target_len)
                training_loss = train(input_variable, lengths, target_variable, mask, max_target_len,  encoder,
                                      decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio,  game_state, delta)

                # Validation set
                valid_in, valid_len, valid_tar, valid_mask, valid_tar_len, valid_game_state, valid_delta = valid_batch
                valid_game_state = valid_game_state.permute(1, 0, 2)
                valid_in, valid_tar, valid_mask, valid_max_tar_len = reshape_text_input(
                    valid_in, valid_tar, valid_mask, valid_tar_len)
                validation_loss = validation(valid_in, valid_len, valid_tar, valid_mask, valid_max_tar_len,
                                             encoder, decoder,  encoder_optimizer, decoder_optimizer, valid_batch_size, valid_game_state, valid_delta)

            elif encoder.use_game_state:

                # Training set
                input_variable, lengths, target_variable, mask, target_len, game_state = train_batch
                input_variable,  target_variable, mask, max_target_len = reshape_text_input(
                    input_variable,  target_variable, mask, target_len)
                game_state = game_state.permute(1, 0, 2)
                training_loss = train(input_variable, lengths, target_variable, mask, max_target_len,  encoder,
                                      decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,  teacher_forcing_ratio,  game_state)
                # Validation set
                valid_in, valid_len, valid_tar, valid_mask, valid_tar_len, valid_game_state = valid_batch
                valid_game_state = valid_game_state.permute(1, 0, 2)
                valid_in, valid_tar, valid_mask, valid_max_tar_len = reshape_text_input(
                    valid_in, valid_tar, valid_mask, valid_tar_len)
                validation_loss = validation(valid_in, valid_len, valid_tar, valid_mask, valid_max_tar_len,
                                             encoder, decoder,  encoder_optimizer, decoder_optimizer, valid_batch_size, valid_game_state)

            elif encoder.use_delta_time:
                # Training set
                input_variable, lengths, target_variable, mask, target_len, delta = train_batch
                input_variable, target_variable, mask, max_target_len = reshape_text_input(
                    input_variable, target_variable, mask, target_len)
                training_loss = train(input_variable, lengths, target_variable, mask, max_target_len,  encoder,
                                      decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,  teacher_forcing_ratio, delta=delta)
                # Validation set
                valid_in, valid_len, valid_tar, valid_mask, valid_tar_len, valid_delta = valid_batch
                valid_in, valid_tar, valid_mask, valid_max_tar_len = reshape_text_input(
                    valid_in, valid_tar, valid_mask, valid_tar_len)
                validation_loss = validation(valid_in, valid_len, valid_tar, valid_mask, valid_max_tar_len,
                                             encoder, decoder,  encoder_optimizer, decoder_optimizer, valid_batch_size, delta=valid_delta)

            else:
                # Training set
                input_variable, lengths, target_variable, mask, target_len = train_batch
                input_variable,  target_variable, mask, max_target_len = reshape_text_input(
                    input_variable,  target_variable, mask, target_len)
                training_loss = train(input_variable, lengths, target_variable, mask, max_target_len,  encoder,
                                      decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,  teacher_forcing_ratio)
                # Validation set
                valid_in, valid_len, valid_tar, valid_mask, valid_tar_len = valid_batch
                valid_in, valid_tar, valid_mask, valid_max_tar_len = reshape_text_input(
                    valid_in, valid_tar, valid_mask, valid_tar_len)
                validation_loss = validation(valid_in, valid_len, valid_tar, valid_mask, valid_max_tar_len,
                                             encoder, decoder,  encoder_optimizer, decoder_optimizer, valid_batch_size)

            # Accumulate losses for this epoch
            print_train_loss += training_loss
            print_valid_loss += validation_loss
            plot_train_loss.append(training_loss)
            plot_valid_loss.append(validation_loss)

        # Print progress
        print_train_loss_avg = print_train_loss / (batch_number+1)
        print_valid_loss_avg = print_valid_loss / (batch_number+1)
        plot_avg_train_loss.append(print_train_loss_avg)
        plot_avg_valid_loss.append(print_valid_loss_avg)
        print("Epoch: {}; Percent complete: {:.1f}%; Average training loss: {:.4f}; Average validation loss: {:.4f}".format(
            epoch, epoch / n_epochs * 100, print_train_loss_avg, print_valid_loss_avg))

        # Save model
        # if epoch % 10 == 0:
        #     name = 'checkpoint epoch:'+str(epoch)
        #     torch.save({
        #         'epoch': epoch,
        #         'encoder_state_dict': encoder.state_dict(),
        #         'decoder_state_dict': decoder.state_dict(),
        #         'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        #         'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        #         'loss': print_train_loss,
        #     }, 'Saved_models/'+name)

        print_train_loss = 0
        print_valid_loss = 0

        # wandb.log({"training loss / epoch": print_train_loss_avg,
        #            'validation loss / epoch': print_valid_loss_avg})

    return plot_train_loss, plot_valid_loss, plot_avg_train_loss, plot_avg_valid_loss


def validation(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
               encoder_optimizer, decoder_optimizer, batch_size, game_tensor=None, delta=None):

    # Ensure eval mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
        if delta is not None:
            delta = delta.to(device)
        if game_tensor is not None:
            game_tensor = game_tensor.to(device)

        # Initialize variables
        loss = 0
        loss_layer = nn.NLLLoss(ignore_index=0)
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        if encoder.use_game_state and encoder.use_delta_time:
            encoder_outputs, encoder_hidden = encoder(
                input_variable, lengths, game_tensor, delta)
        elif encoder.use_game_state:
            encoder_outputs, encoder_hidden = encoder(
                input_variable, lengths, game_tensor)
        elif encoder.use_delta_time:
            encoder_outputs, encoder_hidden = encoder(
                input_variable, lengths, delta=delta)
        else:
            encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = target_variable[0, :]
        decoder_input = decoder_input.unsqueeze(0)

        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        for t in range(max_target_len-1):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss
            mask_loss = loss_layer(decoder_output, target_variable[t+1])
            nTotal = mask.sum()

            # mask_loss, nTotal = maskNLLLoss(
            #     decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return sum(print_losses) / n_totals


def evaluateTestset(encoder, decoder, searcher, voc, test_dataloader, n_eval=10, max_length=50):

    decoded_words = list()
    # Values to revert gamestate to original numbers
    norm = [68844, 66417, 1, 6, 99, 99, 99, 99]

    for _ in range(n_eval):

        # Get next test sample
        test_sample = next(iter(test_dataloader))

        # Extract correct input
        if encoder.use_game_state and encoder.use_delta_time:
            input_variable, lengths, target_variable, mask, target_len, game_state, delta = test_sample
            game_state = game_state.permute(1, 0, 2)
            input_variable, target_variable, mask, target_len = reshape_text_input(
                input_variable, target_variable, mask, target_len)
        elif encoder.use_game_state:
            input_variable, lengths, target_variable, mask, target_len, game_state = test_sample
            game_state = game_state.permute(1, 0, 2)
            input_variable, target_variable, mask, target_len = reshape_text_input(
                input_variable, target_variable, mask, target_len)
        elif encoder.use_delta_time:
            input_variable, lengths, target_variable, mask, target_len, delta = test_sample
            input_variable, target_variable, mask, target_len = reshape_text_input(
                input_variable, target_variable, mask, target_len)
        else:
            input_variable, lengths, target_variable, mask, target_len = test_sample
            input_variable, target_variable, mask, target_len = reshape_text_input(
                input_variable, target_variable, mask, target_len)

        # Convert to text tokens for printing
        input_tokens = [item for sublist in input_variable.tolist()
                        for item in sublist]
        target_tokens = [item for sublist in target_variable.tolist()
                         for item in sublist]
        input_sentence = [voc.index2word[input_tokens[index]]
                          for index in range(lengths)]
        target_sentence = [voc.index2word[target_tokens[index]]
                           for index in range(target_len)]
        input_sentence = ' '.join(input_sentence)
        target_sentence = ' '.join(target_sentence)

        # Print the input/output dialogue and game state
        print('Input: ', input_sentence)
        if encoder.use_game_state:
            final_gamestate = list()
            last_gamestate = game_state[-1].tolist()[0]
            for i, feature in enumerate(last_gamestate):
                final_gamestate.append(round(feature*norm[i]))
                game_state = game_state.to(device)
            print('Last game state: ', final_gamestate)
        print('Target: ', target_sentence)

        # Send to device
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        if encoder.use_delta_time:
            delta = delta.to(device)

        # Decode sentence with searcher
        if encoder.use_delta_time and encoder.use_game_state:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, game_state, delta)
        elif encoder.use_game_state:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, game_state)
        elif encoder.use_delta_time:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, delta=delta)
        else:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable)

        first = True

        # Iterate through predicted tokens and print
        for i, token in enumerate(tokens):
            # If furhats dialogue
            if voc.index2word[target_variable[0].item()] == 'F':
                if first:
                    decoded_words.append('F')
                    first = False
                # If stop token is predicted or max_len reached, break
                if voc.index2word[token.item()] == '/F' or i == len(tokens)-1:
                    decoded_words.append('/F')
                    print('Predicted: ', ' '.join(decoded_words), '\n')
                    decoded_words = list()
                    break
                else:
                    decoded_words.append(voc.index2word[token.item()])
            # If humans dialogue
            else:
                if first:
                    decoded_words.append('H')
                    first = False
                # If stop token is predicted or max_len reached, break
                if voc.index2word[token.item()] == '/H' or i == len(tokens)-1:
                    decoded_words.append('/H')
                    print('Predicted: ', ' '.join(decoded_words), '\n')
                    decoded_words = list()
                    break
                else:
                    decoded_words.append(voc.index2word[token.item()])


def specific_eval(encoder, decoder, searcher, voc, test_cases,  max_length=50):

    decoded_words = list()
    norm = [68844, 66417, 1, 6, 99, 99, 99, 99]
    for test_case in test_cases:

        if encoder.use_game_state and encoder.use_delta_time:
            input_variable, lengths, target_variable, mask, target_len, game_state, delta = test_case
            game_state = game_state.unsqueeze(1)
            delta = delta.unsqueeze(0)
            delta = delta.to(device)
        elif encoder.use_game_state:
            input_variable, lengths, target_variable, mask, target_len, game_state = test_case
            game_state = game_state.unsqueeze(1)
        elif encoder.use_delta_time:
            input_variable, lengths, target_variable, mask, target_len, delta = test_case
            delta = delta.unsqueeze(0)
            delta = delta.to(device)
        else:
            input_variable, lengths, target_variable, mask, target_len = test_case

        input_variable = input_variable.unsqueeze(1)
        target_variable = target_variable.unsqueeze(1)
        mask = mask.unsqueeze(1)
        lengths = lengths.unsqueeze(0)
        target_len = target_len.unsqueeze(0)
        max_target_len = max(target_len)

        input_tokens = [item for sublist in input_variable.tolist()
                        for item in sublist]
        target_tokens = [item for sublist in target_variable.tolist()
                         for item in sublist]
        input_sentence = [voc.index2word[input_tokens[index]]
                          for index in range(lengths)]
        target_sentence = [voc.index2word[target_tokens[index]]
                           for index in range(target_len)]
        input_sentence = ' '.join(input_sentence)
        target_sentence = ' '.join(target_sentence)

        print('Input: ', input_sentence)

        if encoder.use_game_state:
            final_gamestate = list()
            last_gamestate = game_state[-1].tolist()[0]
            for i, feature in enumerate(last_gamestate):
                final_gamestate.append(round(feature*norm[i]))
            print('Last game state: ', final_gamestate)
            game_state = game_state.to(device)

        print('Target: ', target_sentence)

        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)

        # Decode sentence with searcher
        if encoder.use_delta_time and encoder.use_game_state:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, game_state, delta)
        elif encoder.use_game_state:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, game_state)
        elif encoder.use_delta_time:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable, delta=delta)
        else:
            tokens, scores = searcher(
                input_variable, lengths, max_length, target_variable)

        first = True
        for i, token in enumerate(tokens):

            if voc.index2word[target_variable[0].item()] == 'F':
                if first:
                    decoded_words.append('F')
                    first = False
                if voc.index2word[token.item()] == '/F' or i == len(tokens)-1:
                    decoded_words.append('/F')
                    print('Predicted: ', ' '.join(decoded_words), '\n')
                    decoded_words = list()
                    break
                else:
                    decoded_words.append(voc.index2word[token.item()])

            else:
                if first:
                    decoded_words.append('H')
                    first = False
                if voc.index2word[token.item()] == '/H' or i == len(tokens)-1:
                    decoded_words.append('/H')
                    print('Predicted: ', ' '.join(decoded_words), '\n')
                    decoded_words = list()
                    break
                else:
                    decoded_words.append(voc.index2word[token.item()])
