# master_thesis_seq2seq

## Getting started

### 1. Clone repo

### 2. run parse_raw_data.py

Since the complete dataset is too large to upload to github this will parse		
the raw data and create the full datasets to be used for training. Syncing the gamestate data may take a little while. 

### 3. open main.py


To configure the input modalities, model architecture, training/optimization and evaluation there are four clearly maked sections at the begining of the scrpit to do so.

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

- hidden_size = 50
- encoder_n_layers = 1
- decoder_n_layers = 1
- dropout = 0
- batch_size = 10
- load_embeddings = False     
- bi_directional = True 

#### Configure training/optimization
#### Configure Evaluation
