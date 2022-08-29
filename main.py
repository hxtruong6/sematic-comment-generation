import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, ldamodel
import pandas as pd

import configparser
import argparse


def read_ini(file_path, section='APP'):
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        for key in config[section]:
            print((key, config[section][key]))
    return config[section]


def build_corpus(df: pd.DataFrame):
    try:
        # [print(sentences.split()) for sentences in df['review_clean']]

        word_corpus = [str(sentences).split() for sentences in df['review_clean']]
        # Creating Document Term Matrix
        id2word = corpora.Dictionary(word_corpus)
        # Converting list of documents (corpus) into Document Term Matrix using the dictionary
        corpus = [id2word.doc2bow(text) for text in word_corpus]

        return {'word_corpus': word_corpus, 'id2word': id2word, 'corpus': corpus}
    except Exception as e:
        print(e)


def lda_training(corpus, id2word, num_topics: int, epoches: int = 10, chunksize: int = 100,
                 per_word_topics: bool = True, distributed: bool = False):
    lda_model = ldamodel.LdaModel(corpus=corpus,
                                  id2word=id2word,
                                  num_topics=num_topics,
                                  random_state=100,
                                  update_every=1,
                                  chunksize=chunksize,
                                  passes=epoches,
                                  alpha='auto',
                                  per_word_topics=per_word_topics,
                                  distributed=distributed
                                  )
    return lda_model


def evaluate_lda(lda_model: ldamodel, corpus, word_corpus, id2word):
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    cm = CoherenceModel(model=lda_model, texts=word_corpus, dictionary=id2word, coherence='c_v')

    # TODO: error here
    coherence_lda = cm.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def get_configs(init_file=None):
    if init_file:
        config = read_ini('config.ini')
        # print(f"Config: {config}")

        # Configurate variables
        num_topics = int(config['NUM_TOPIC'])
        num_words = int(config['NUM_WORDS'])
        epoches = int(config['EPOCHES'])
        return [num_topics, num_words, epoches]
    else:
        # Create the parser
        parser = argparse.ArgumentParser()
        # Add an argument
        parser.add_argument('--num_topic', type=int, required=True)
        parser.add_argument('--num_words', type=int, required=False, default=10)
        parser.add_argument('--epoches', type=int, required=True)

        # Parse the argument
        args = parser.parse_args()
        # Print "Hello" + the user input argument
        return [args.num_topic, args.num_words, args.epoches]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # init_file = 'config.ini'
    init_file = None  # TODO: uncomment when training

    num_topics, num_words, epoches = get_configs(init_file)

    # --------
    df = pd.read_csv("dataset/tripadvisor_clean_reviews.csv")
    df = df[:100]
    df['review_clean'].astype(dtype=str, copy=False)
    # print(df['review_clean'].head())

    builded_corpus = build_corpus(df)
    # print(builded_corpus)
    word_corpus = builded_corpus['word_corpus']
    id2word = builded_corpus['id2word']
    corpus = builded_corpus['corpus']

    lda_model = lda_training(corpus=corpus, id2word=id2word, num_topics=num_topics, epoches=epoches)
    [print(topics) for topics in lda_model.print_topics(num_topics=num_topics, num_words=num_words)]
