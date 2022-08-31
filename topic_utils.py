import gensim.corpora as corpora
import pandas as pd
from gensim.models import CoherenceModel, ldamodel


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
    cm = CoherenceModel(model=lda_model, texts=word_corpus, dictionary=id2word, coherence='c_v')

    eval_lda = {
        'perplexity': lda_model.log_perplexity(corpus),
        'coherence_score': cm.get_coherence()
    }
    # Compute Perplexity
    print('Perplexity: ', eval_lda['perplexity'])  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    print(f"Coherence Score: {eval_lda['coherence_score']}\n")

    return eval_lda
