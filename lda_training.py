import pandas
import pandas as pd

import pandas_parallel
from topic_utils import build_corpus, lda_training, evaluate_lda
from utils import write2file, get_configs
from datetime import datetime
from os import makedirs, path
from gensim.test.utils import datapath


# now = datetime.now()  # current date and time
# date_time = now.strftime("%Y%m%d-%H%M%S")
# makedirs(f'results/{date_time}', exist_ok=True)
# prefix_folder = f'results/{date_time}'

def create_result_df(file_path='results/result_df.csv'):
    if path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        makedirs("results", exist_ok=True)
        df = pd.DataFrame(
            {'hotel': [], 'hotel_name': [], 'created_at': [],
             'num_topics': [], 'num_words': [], 'epoches': [],
             'topics': [], 'perplexity': [], 'coherence_score': []})
        return df


result_df = create_result_df()
configs = None


def handle_topic_df(df: pandas.DataFrame, configs):
    builded_corpus = build_corpus(df)
    # print(builded_corpus)
    word_corpus = builded_corpus['word_corpus']
    id2word = builded_corpus['id2word']
    corpus = builded_corpus['corpus']

    # print(configs)

    lda_model = lda_training(corpus=corpus, id2word=id2word, num_topics=configs['num_topics'],
                             epoches=configs['epoches'])

    # is_global = "GLOBAL_" if configs['global_topic'] else ""
    # temp_model = datapath(f"./results/{is_global}model_{configs['num_topics']}_{configs['epoches']}.model")
    # lda_model.save(temp_model)

    topics = [topics for topics in
              lda_model.print_topics(num_topics=configs['num_topics'], num_words=configs['num_words'])]

    eval = evaluate_lda(lda_model=lda_model, corpus=corpus, word_corpus=word_corpus, id2word=id2word)

    return {'topics': '\n'.join([str(topic) for topic in topics]), 'eval': eval}

    # if writing_file:
    #     write2file(writing_file, '\n'.join([str(topic) for topic in topics]))


def handle_global_topic(df: pandas.DataFrame):
    global result_df
    global configs

    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    res = handle_topic_df(df, configs)
    result_df = insert_to_result_df(result_df, -1, "GLOBAL_TOPIC", date_time, configs, res)


def handle_hotel_comments(df: pandas.DataFrame):
    global result_df
    global configs

    grouped_df = df.groupby(['hotel', 'hotel_name'])

    for name, subgroup_df in grouped_df:
        date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print(f"Hotel: {name}")
        res = handle_topic_df(subgroup_df, configs)
        result_df = insert_to_result_df(result_df, name[0], name[1], date_time, configs, res)


def preprocessing(file_path: str):
    df = pd.read_csv(file_path)
    df = df[:10]
    df['review_clean'].astype(dtype=str, copy=False)
    return df


def save_result_df(df: pd.DataFrame, file_path='results/result_df.csv'):
    df.to_csv(file_path, index=False)


def insert_to_result_df(result_df, hotel, hotel_name, date_time, configs, res):
    return pd.concat([result_df, pd.DataFrame([{
        'hotel': int(hotel), 'hotel_name': hotel_name, 'created_at': date_time,
        'num_topics': configs['num_topics'], 'num_words': configs['num_words'], 'epoches': configs['epoches'],
        'topics': res['topics'], 'perplexity': res['eval']['perplexity'],
        'coherence_score': res['eval']['coherence_score']
    }])])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # init_file = 'config.ini'
    init_file = None  # TODO: uncomment when training

    configs = get_configs(init_file)
    print(f"configs: {configs}")

    # --------
    df = preprocessing("dataset/tripadvisor_clean.csv")

    if configs['global_topic']:
        handle_global_topic(df)
    else:
        handle_hotel_comments(df)

    # print(result_df.info())
    save_result_df(result_df)

    # df1 = create_result_df()
    # print(df1)
