# master_thesis_seq2seq

## Getting started

### 1. Clone repo

### 2. run parse_raw_data.py

Since the complete dataset is too large to upload to github this will parse		
the raw data and create the full datasets to be used for training. Syncing the gamestate data may take a little while. 

### 3. Configure model/open main.py


To configure the input modalities, model architecture, training/optimization and evaluation there are four clearly maked sections at the begining of the script to do so.

#### Configure data structure
This is the most impactful section as it controls the input shapes and modalities which greatly effects the model configuration. It has the following boolean options:

- use_dialogue_history
- use_game_state 
- use_delta_time 
- game_only 

If all are set to False the network will only use dialogue data and only use the previous utterance as input to predict the next one.

To include more dialogue history (three last utterances) in the input set use_dialogue_history to True.

To include the gamestate data set use_game_state to True. 

To include the time between the input utterances and the target set use_delta_time to True. 

If you only want to have the gamestate as input set game_only to True (This will automatically set use_dialogue_history and use_delta_time to False and set use_game_state to True).


#### Configure models
The next section is about the models and includes the following options:

- hidden_size: hidden dimension of both dialogue and gamestate GRU in encoder/decoder
- encoder_n_layers: number of layers for both dialogue and gamestate GRU in encoder 
- decoder_n_layers:  number of layers in decoder GRU
- dropout: dropout in all GRU-units (if n_layers = 1 dropout defaults to 0)
- batch_size: size of training batches
- load_embeddings: Loads embeding weights from GLoVe if True    
- bi_directional: Sets dialogue GRU in encoder to bidirectional if True  

#### Configure training/optimization
- clip: Gradient clipping threshold
- teacher_forcing_ratio: Probability of using teacher forcing at any given time [0,1]
- learning_rate: The learning rate used during optimization.
- decoder_learning_ratio: A factor placed on the learning rate of the decoders learning rate, encoder/decoder has same lr if this is 1. 
- n_epochs: Number of epoch to train
- use_teacher_decay: Uses linear decay for teacher forcing if True, default setting is 1 to 0 over 70% of the epochs. You can find the function teacher_decay in utils.py to change this. 

#### Configure Evaluation
- evaluate_on_test: Randomly evaluates on a given amount of samples from the test set. 
- n_eval: Number of test samples to evaluate
- evaluate_specific: Evaluates on specific samples of snakes, ladders and wins if True.
- plot_embeddings: Plots the tsne figures of the word embeddings for the 100 most frequent words if True. Also gives the 3 closest neighbours to snake and ladder.


### 4. Run main.py

