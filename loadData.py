import string
import pandas as pd
import os


def load_dialogue(filename, player_token):
    """

    Function that parses the raw dialogue data, handles exceptions such as nan
    and return a cleaned data frame. Also adds player tokens if player_token is True.

    """

    data = []
    df = pd.read_csv(filename)

    if player_token:
        for index, row in df.iterrows():
            if isinstance(row['Annotations - Furhat'], str):
                clean_text = row['Annotations - Furhat']
                data.append([row['Begin Time - msec'],
                             'F '+clean_text.lower()+' /F '])
            if isinstance(row['Annotations - Human'], str):
                clean_text = row['Annotations - Human']
                data.append([row['Begin Time - msec'],
                             'H '+clean_text.lower()+' /H '])
        df = pd.DataFrame(data, columns=['Time', 'Dialogue'])
    else:
        for index, row in df.iterrows():
            if isinstance(row['Annotations - Furhat'], str):
                clean_text = row['Annotations - Furhat']
                data.append([row['Begin Time - msec'], clean_text.lower()])

            if isinstance(row['Annotations - Human'], str):
                clean_text = row['Annotations - Human']
                data.append([row['Begin Time - msec'], clean_text.lower()])
        df = pd.DataFrame(data, columns=['Time', 'Dialogue'])

    return df


def clean_dialogue(player_token):
    TextFileNames = ['Daniel.csv', 'Jenny.csv', 'Julia.csv', 'Lukas.csv', 'Martti.csv', 'Max.csv',
                     'Michaela.csv', 'Olle.csv', 'Pernilla.csv', 'Sebastian.csv', 'Teo.csv', 'Tim.csv']

    load_dir = 'raw_dialogue/'
    save_dir = 'dialogue_clean/'

    train_list = list()
    for path in TextFileNames:
        trainingData = load_dialogue(load_dir+path, player_token)
        try:
            trainingData.to_csv(save_dir+path, index=False, header=True)
        except:
            os.mkdir(save_dir)
            trainingData.to_csv(save_dir+path, index=False, header=True)
