import pandas as pd
import numpy as np
import os

text_file_names = ['Daniel.csv', 'Jenny.csv', 'Julia.csv', 'Lukas.csv', 'Max.csv',
                   'Olle.csv', 'Pernilla.csv', 'Sebastian.csv', 'Teo.csv', 'Tim.csv', 'Michaela.csv', 'Martti.csv']
game_file_names = ['GameState_25_DANIEL.csv', 'GameState_30_JENNY.csv', 'GameState_26_JULIA.csv', 'GameState_21_LUKAS.csv', 'GameState_23_MAX.csv', 'GameState_22_OLLE.csv',
                   'GameState_24_PERNILLA.csv', 'GameState_23_SEBASTIAN.csv', 'GameState_28_TEO.csv', 'GameState_19_TIM.csv', 'GameState_28_MICHAELA.csv', 'GameState_22_MARTII_2.csv']

load_dir_game = 'raw_gamestate/'
load_dir_text = 'dialogue_clean/'
save_dir = 'gamestate_datasets/'


def load_gamestate(filename):
    df = pd.read_csv(load_dir_game+filename)
    df = df.drop(['UniversalTime', 'FrameNr', 'Unnamed: 0'], axis=1)
    return df


def sync_game_state(text_data, game_data, zero_dim):

    synced_game_state = list()
    game_state = list()

    for text_index, text_row in text_data.iterrows():
        current_time = text_row['Time']
        for game_index, game_row in game_data.iterrows():
            if game_row['EntireGameTime'] > current_time or game_index == len(game_data)-1:
                synced_game_state.append(game_state)
                game_state = list()
                break
            else:
                game_state.append([game_row['TurnTime']/68844, game_row['Duration']/66417, game_row['WhosTurn'], game_row['DiceValue']/6, game_row['Human']/99,
                                   game_row['Furhat']/99, game_row['ifSnake']/99, game_row['ifLadder']/99])

    return synced_game_state


def zero_padding(seq, max_len, zero_dim):
    zero_pad = [0]*zero_dim
    for _ in range((max_len-len(seq))):
        seq.insert(0, zero_pad)
    return seq


def parse_game_state():

    text = [pd.read_csv(load_dir_text+fileName)
            for fileName in text_file_names]
    game_states = [load_gamestate(path) for path in game_file_names]

    max_dur = 66417
    max_turn = 68844
    zero_dim = 8
    max_length = -1
    games_synced = list()

   # Training set
    for i, name in enumerate(text_file_names):
        print('suncing gamestate for participant: ', name)
        synced_game = sync_game_state(text[i], game_states[i], zero_dim)
        if len(synced_game[-1]) > max_length:
            max_length = len(synced_game[-1])
        assert(len(synced_game) == len(text[i]))
        games_synced.extend(synced_game)

    print('The longest game sequence is: ', max_length)

    for i, seq in enumerate(games_synced):
        games_synced[i] = zero_padding(seq, max_length, zero_dim)

    np_full = np.array(games_synced)
    np_full = np_full.reshape(-1, zero_dim)
    df_full = pd.DataFrame(np_full)

    try:
        df_full.to_csv(save_dir+'game_state.csv',
                       index=False, header=True)
    except:
        os.mkdir(save_dir)
        df_full.to_csv(save_dir+'game_state.csv',
                       index=False, header=True)
