import pandas
import pandas as pd

from topic_utils import build_corpus, lda_training, evaluate_lda
from utils import write2file, get_configs
from datetime import datetime
from os import makedirs

now = datetime.now()  # current date and time
date_time = now.strftime("%Y%m%d-%H%M%S")
makedirs(f'results/{date_time}', exist_ok=True)
prefix_folder = f'results/{date_time}'


def handle_topic_df(df: pandas.DataFrame, configs, writing_file=None):
    builded_corpus = build_corpus(df)
    # print(builded_corpus)
    word_corpus = builded_corpus['word_corpus']
    id2word = builded_corpus['id2word']
    corpus = builded_corpus['corpus']

    # print(configs)

    lda_model = lda_training(corpus=corpus, id2word=id2word, num_topics=configs['num_topics'],
                             epoches=configs['epoches'])
    topics = [topics for topics in
              lda_model.print_topics(num_topics=configs['num_topics'], num_words=configs['num_words'])]

    evaluate_lda(lda_model=lda_model, corpus=corpus, word_corpus=word_corpus, id2word=id2word)

    if writing_file:
        write2file(writing_file, '\n'.join([str(topic) for topic in topics]))


def handle_global_topic(df: pandas.DataFrame, configs):
    handle_topic_df(df, configs, writing_file=f'{prefix_folder}/global_topics.txt')


def handle_hotel_comments(df: pandas.DataFrame, configs):
    grouped_df = df.groupby(['hotel', 'hotel_name'])

    for name, subgroup_df in grouped_df:
        print(f"Hotel: {name}")
        # print(group['review_clean'])
        writing_file = f"{prefix_folder}/{str(name[0]).lower()}_topics.txt"
        # print(writing_file)
        handle_topic_df(subgroup_df, configs, writing_file)


def preprocessing(file_path: str):
    df = pd.read_csv(file_path)
    df = df[:1000]
    df['review_clean'].astype(dtype=str, copy=False)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # init_file = 'config.ini'
    init_file = None  # TODO: uncomment when training
    configs = get_configs(init_file)
    print(f"configs: {configs}")

    # --------
    df = preprocessing("dataset/tripadvisor_clean.csv")
    if configs['global_topic']:
        handle_global_topic(df, configs)
    else:
        handle_hotel_comments(df, configs)
