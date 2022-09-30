import pandas
import pandas as pd

import pandas_parallel
from topic_utils import build_corpus, lda_training, evaluate_lda
from utils import write2file, get_configs
from datetime import datetime
from os import makedirs, path
import time

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
configs = {}


def handle_topic_df(df: pandas.DataFrame, configs, builded_corpus=None):
    print(f"handle_topic_df: {configs}")

    if builded_corpus is None:
        start_time = time.time()
        builded_corpus = build_corpus(df)
        print("--- Builded_corpus: %s seconds ---" %
              (round(time.time() - start_time)))
    # print(builded_corpus)
    word_corpus = builded_corpus['word_corpus']
    id2word = builded_corpus['id2word']
    corpus = builded_corpus['corpus']

    # print(configs)

    lda_model = lda_training(corpus=corpus, id2word=id2word, num_topics=configs['num_topics'],
                             epoches=configs['epoches'])

    eval = evaluate_lda(lda_model=lda_model, corpus=corpus,
                        word_corpus=word_corpus, id2word=id2word)

    return lda_model, eval


def handle_saving(lda_model, eval, num_words, hotel, hotel_name, date_time):
    global result_df
    global configs

    topics = [topics for topics in
              lda_model.print_topics(num_topics=configs['num_topics'], num_words=num_words)]

    topics_str = '\n'.join([str(topic) for topic in topics])
    # result_df = insert_to_result_df(result_df, hotel, hotel_name, date_time, configs, eval, topics_str)

    result_df = pd.concat([result_df, pd.DataFrame([{
        'hotel': int(hotel), 'hotel_name': hotel_name, 'created_at': date_time,
        'num_topics': configs['num_topics'], 'num_words': num_words, 'epoches': configs['epoches'],
        'topics': topics_str, 'perplexity': eval['perplexity'],
        'coherence_score': eval['coherence_score']
    }])])


def handle_global_topic(df: pandas.DataFrame, builded_corpus=None, num_words_list=[]):
    global result_df
    global configs

    if num_words_list is None:
        num_words_list = [10]

    date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    lda_model, eval = handle_topic_df(df, configs, builded_corpus)

    for num_words in num_words_list:
        print(f"Saving (num_words={num_words})...")
        handle_saving(lda_model, eval, num_words, -
                      1, "GLOBAL_TOPIC", date_time)


def handle_hotel_comments(df: pandas.DataFrame, num_words_list=None):
    if num_words_list is None:
        num_words_list = [10]

    global result_df
    global configs

    grouped_df = df.groupby(['hotel', 'hotel_name'])

    for name, subgroup_df in grouped_df:
        date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        print(f"Hotel: {name}")
        lda_model, eval = handle_topic_df(subgroup_df, configs)

        for num_words in num_words_list:
            print(f"Saving (num_words={num_words})...")
            handle_saving(lda_model, eval, num_words,
                          name[0], name[1], date_time)
        print("------****------\n")


def preprocessing(file_path: str):
    df = pd.read_csv(file_path)
    # df = df[:10000]
    df['review_clean'].astype(dtype=str, copy=False)
    return df


def save_result_df(df: pd.DataFrame, file_path='results/result_df.csv'):
    df.to_csv(file_path, index=False)
    print("Saved!")


def train(num_topic_cf, num_words_cf, epoches_cf, global_topic=True):
    global configs

    df = preprocessing("dataset/tripadvisor_clean.csv")

    builded_corpus = None
    if global_topic:
        start_time = time.time()
        builded_corpus = build_corpus(df)
        print("---Builded_corpus: %s seconds ---" %
              (round(time.time() - start_time, 2)))

    num_words_list = [num_words for num_words in range(
        num_words_cf[0], num_words_cf[1])]

    for num_topic in range(num_topic_cf[0], num_topic_cf[1]):
        for epoches in epoches_cf:
            start_time = time.time()
            # ----
            configs['num_topics'] = num_topic
            configs['epoches'] = epoches

            if global_topic:
                handle_global_topic(df, builded_corpus, num_words_list)
            else:
                handle_hotel_comments(df, num_words_list)

            save_result_df(result_df)
            # ----
            print("---Train: %s seconds ---\n" %
                  (round(time.time() - start_time, 2)))


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
