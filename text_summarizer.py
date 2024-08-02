import gensim.downloader
from rouge import Rouge
import tensorflow_datasets as tfds
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity



datasets = {
    'cnn_dailymail': tfds.load('cnn_dailymail', split='test', as_supervised=True, shuffle_files=False),
    'gigaword': tfds.load('gigaword', split='test', as_supervised=True, shuffle_files=False),
}

def overall_score(text, vectorizer):
    sim_matrix = similarity_matrix(text, vectorizer)
    effective_lens = [effective_length(s) for s in split_into_sentences(text)]
    real_lens = [real_length(s) for s in split_into_sentences(text)]

    rep_score = np.log(represent_score_(sim_matrix, 0.22))
    div_score = np.log(diversity_score_(sim_matrix, rep_score))
    len_score = np.log(length_score_(effective_lens, real_lens))
    score = rep_score + div_score + len_score
    return [(sent, s) for sent, s in zip(split_into_sentences(text), score)]


def represent_score_(sim_matrix, delta):
    sentences_count = len(sim_matrix)
    result = np.zeros(shape=(sentences_count,))
    for i in range(sentences_count):
        a = (np.array(sim_matrix[i]) - delta) > 0
        t = np.ma.array(a * 1.0, mask=False)
        t.mask[i] = True
        result[i] = np.sum(t) / sentences_count
        if not result[i]:
            result[i] = 10**(-10)
    return result


def diversity_score_(sim_matrix, s_rep_vector):
    sim_matrix = np.array(sim_matrix)
    s_rep_vector = np.array(s_rep_vector)

    sentences_count = len(sim_matrix)
    result = np.zeros(shape=(sentences_count,))
    for i in range(sentences_count):
        mask = (s_rep_vector > s_rep_vector[i]) * 1
        if np.any(mask):
            result[i] = 1 - np.max(sim_matrix[i] * mask)
        else:
            t = np.ma.array(sim_matrix[i], mask=False)
            t.mask[i] = True
            result[i] = 1 - np.min(t)
        if not result[i]:
            result[i] = 10**(-10)
    return result


def length_score_(effective_lens, real_lens):
    sentences_count = len(effective_lens)
    result = np.zeros(shape=(sentences_count,))
    for i in range(sentences_count):
        if real_lens[i]:
            result[i] = (effective_lens[i] / np.max(effective_lens)) * (np.max(real_lens) / real_lens[i])
        if not result[i]:
            result[i] = 10**(-10)
    return result


def split_into_sentences(text):
    return list(map(lambda x: x.strip(), re.split(r'; |\n|\. ', text)))


def effective_length(sentence):
    stop_words = set(stopwords.words(['russian', 'english']))
    unique_words = set()
    for word in word_tokenize(sentence):
        if word not in stop_words:
            unique_words.add(word)
    return len(unique_words)


def real_length(sentence):
    return len(word_tokenize(sentence))


def similarity_matrix(text, vectorizer):
    sentences = split_into_sentences(text)
    vec_sentences = [word2vec_encode_text(s, vectorizer) for s in sentences]
    return np.array(cosine_similarity(vec_sentences), dtype=float)


def word2vec_encode_text(text, word2vec):
    tokenizer = re.compile(r'\b[\w./-]+\b', re.I)
    if len(tokenizer.findall(text)) == 0:
        return np.zeros(word2vec.vector_size)
    return np.array([
        word2vec.get_vector(x) if x in word2vec.vocab else np.zeros(word2vec.vector_size)
        for x in tokenizer.findall(text)
    ]).mean(axis=0)


def select_top(sentences_with_score, top=10):
    score = list(map(lambda x: (x[0], -np.inf) if np.isnan(x[1]) else x, sentences_with_score))
    sent_with_order = [(s[0], s[1], i) for i, s in enumerate(score)]
    sorted_s = sorted(sent_with_order, key=lambda x: -x[1])[:top]
    return [s[0] for s in sorted(sorted_s, key=lambda x: x[2])]



def summarize(x, vectors):
    scores = overall_score(x, vectors)
    return '. '.join(select_top(scores, 10))



