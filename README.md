# master_thesis_seq2seq

## Getting started

### 1. Clone repo

### 2. run parse_raw_data.py

Since the complete dataset is too large to upload to github as one file this will parse		
the raw data and create the full datasets to be used for training. Syncing the gamestate data may take a little while. 

#### If you want to you should now already be able to run the model in it's simplest form by running main.py. 


### 3. Install requirements.txt

### 4. Configure model - main.py

To configure the input modalities, model architecture, training/optimization and evaluation there are four clearly maked sections at the begining of the script to do so.

![config_model](https://user-images.githubusercontent.com/55242743/117950460-7eeace00-b313-11eb-9f2b-9dda3ff338af.png)


#### Configure data structure
This section controls the input shapes and modalities which greatly effects the model configuration. It has the following boolean options:

- use_dialogue_history: dialogue input is 3 utterances if True otherwise 1.
- use_game_state: includes gamestate dataset and GRu if True
- use_delta_time: includes the time between the input utterances and the target if True
- game_only: includes only gamestate data in model if True, this overrides the other options if set to True


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


### 5. Run main.py

## Structure of repo
Here is an overview of the contents of the scripts and their functions

#### main.py 
High level comands, such as training and evaluation. 
Configure model and input data. 

#### encoder_decoder.py
Seq2seq model contains encoder/decoder class, greedysearchdecoder, train, validation and evaluation functions 

#### vocab.py
Contains the vocabulary class

#### utils.py
Contains utility functions for: padding, masking, createing tensor dataset from batch of samples, plotting tsne figures, teacher decay. 


