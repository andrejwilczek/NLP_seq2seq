import pandas as pd
import os


def create_subset(sentences, history, eos_tag):

    dataset = list()
    three_prev_sentence = 'EOS '
    two_prev_sentence = 'EOS '
    prev_sentence = 'EOS '

    if history:
        if eos_tag:
            for sentence in sentences:
                sentence = sentence.replace(".", " . EOS").replace(
                    ",", " , ").replace("?", " ? EOS").replace("!", " ! EOS")

                input_seq = three_prev_sentence + \
                    two_prev_sentence + prev_sentence
                target_seq = sentence

                input_seq = " ".join(input_seq.split())
                target_seq = " ".join(target_seq.split())

                dataset.append([input_seq, target_seq])
                three_prev_sentence = two_prev_sentence
                two_prev_sentence = prev_sentence
                prev_sentence = sentence
        else:
            for sentence in sentences:
                sentence = sentence.replace(".", " . ").replace(
                    ",", " , ").replace("?", " ? ").replace("!", " ! ")

                input_seq = three_prev_sentence + \
                    two_prev_sentence + prev_sentence
                target_seq = sentence

                input_seq = " ".join(input_seq.split())
                target_seq = " ".join(target_seq.split())

                dataset.append([input_seq, target_seq])
                three_prev_sentence = two_prev_sentence
                two_prev_sentence = prev_sentence
                prev_sentence = sentence
    else:
        if eos_tag:
            for sentence in sentences:
                sentence = sentence.replace(".", " . EOS").replace(
                    ",", " ,").replace("?", " ? EOS").replace("!", " ! EOS")
                sentence = " ".join(sentence.split())

                dataset.append([prev_sentence, sentence])
                prev_sentence = sentence

        else:
            for sentence in sentences:
                sentence = sentence.replace(".", " . ").replace(
                    ",", " ,").replace("?", " ? ").replace("!", " ! ")
                sentence = " ".join(sentence.split())

                dataset.append([prev_sentence, sentence])
                prev_sentence = sentence
    return dataset


def create_deltas(times, history):

    if history:
        deltas = list()
        three_prev_time = 0
        two_prev_time = 0
        prev_time = 0
        for time in times:
            delta1 = (time-prev_time)/(60000)
            delta2 = (time-two_prev_time)/(60000)
            delta3 = (time-three_prev_time)/(60000)
            deltas.append([delta1, delta2, delta3])
            three_prev_time = two_prev_time
            two_prev_time = prev_time
            prev_time = time

    else:
        deltas = list()
        prev_time = 0
        for time in times:
            delta = (time-prev_time)/60000
            deltas.append(delta)
            prev_time = time

    return deltas


def create_dialogue_dataset(file_names, history, load_dir, save_dir, eos_tag):

    full_dataset = list()
    full_deltas = list()

    for path in file_names:
        df_all = pd.read_csv(load_dir+path)
        sentences = df_all['Dialogue']
        times = df_all['Time']
        all_data = create_subset(sentences, history, eos_tag)
        delta_time = create_deltas(times, history)
        full_deltas.extend(delta_time)
        full_dataset.extend(all_data)

    assert(len(full_dataset) == len(full_deltas))
    df_all = pd.DataFrame(full_dataset, columns=['input', 'target'])

    # Save data to file
    try:
        if history:
            df_deltas = pd.DataFrame(full_deltas, columns=[
                'delta1', 'delta2', 'delta3'])
            df_deltas.to_csv(save_dir+'delta_time_history.csv',
                             index=False, header=True)
            df_all.to_csv(save_dir+'dialogue_history.csv',
                          index=False, header=True)
        else:
            df_deltas = pd.DataFrame(full_deltas, columns=['delta'])
            df_deltas.to_csv(save_dir+'delta_time.csv',
                             index=False, header=True)
            df_all.to_csv(save_dir+'dialogue.csv', index=False, header=True)
    except:
        os.mkdir(save_dir)
        if history:
            df_deltas = pd.DataFrame(full_deltas, columns=[
                'delta1', 'delta2', 'delta3'])
            df_deltas.to_csv(save_dir+'delta_time_history.csv',
                             index=False, header=True)
            df_all.to_csv(save_dir+'dialogue_history.csv',
                          index=False, header=True)
        else:
            df_deltas = pd.DataFrame(full_deltas, columns=['delta'])
            df_deltas.to_csv(save_dir+'delta_time.csv',
                             index=False, header=True)
            df_all.to_csv(save_dir+'dialogue.csv', index=False, header=True)


def parse_dialogue(history, eos_tag):

    # File names
    file_names = ['Daniel.csv', 'Jenny.csv', 'Julia.csv', 'Lukas.csv', 'Max.csv',
                  'Olle.csv', 'Pernilla.csv', 'Sebastian.csv', 'Teo.csv', 'Tim.csv', 'Michaela.csv', 'Martti.csv']

    # Directory names
    save_dir = 'dialogue_datasets/'
    load_dir = 'dialogue_clean/'

    if history:
        create_dialogue_dataset(file_names, history,
                                load_dir, save_dir, eos_tag)
    else:
        create_dialogue_dataset(file_names, history,
                                load_dir, save_dir, eos_tag)
