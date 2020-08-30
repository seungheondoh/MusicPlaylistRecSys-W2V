import re
import os
import io
import json
import distutils.dir_util
import pickle
import numpy as np
from collections import Counter
from numpy import dot, float32 as REAL, array, ndarray, sum as np_sum, prod
from six import string_types, integer_types
from gensim import matutils


def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./data/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)

def vector_most_similar(self, kmeans, all_words, topn, restrict_vocab=None):
    self.init_sims()
    mean = matutils.unitvec(kmeans).astype(REAL)
    limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[restrict_vocab]
    dists = dot(limited, mean)
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    if restrict_vocab:
        result = [(self.index2word[restrict_vocab[sim]], float(dists[sim])) for sim in best if restrict_vocab[sim] not in all_words]
    else:
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def entity_most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, cos_mul=False):
    if positive is None:
        positive = []
    if negative is None:
        negative = []
    self.init_sims()

    if isinstance(positive, string_types) and not negative:
        positive = [positive]

    if cos_mul == True:
        all_words = {
        self.vocab[word].index for word in positive + negative
        if not isinstance(word, ndarray) and word in self.vocab
        }
        positive = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in positive
        ]
        negative = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in negative
        ]
        if not positive:
            raise ValueError("cannot compute similarity with no input")
        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        if restrict_vocab:
            pos_dists = [((1 + dot(self.vectors_norm[restrict_vocab], term)) / 2) for term in positive] 
        else:
            pos_dists = [((1 + dot(self.vectors_norm, term)) / 2) for term in positive]
        dists = prod(pos_dists, axis=0)
    else:
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]
        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.word_vec(word, use_norm=True))
                if word in self.vocab:
                    all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[restrict_vocab]
        dists = dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    if restrict_vocab:
        result = [(self.index2word[restrict_vocab[sim]], float(dists[sim])) for sim in best if restrict_vocab[sim] not in all_words]
    else:
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def cleanText(sent):
    result = re.sub('[-=+,#/\?:;^$@*\"※~%ㆍ!』\\‘|\(\)\[\]\<\>`\'》]', '', sent)
    result_with_space = re.sub('[.…]', '', result)
    return result_with_space.lower()

def _get_candidate_set(songs_in_dictionary,filtered_song, threshold, topk, single_query=1):
    mid_list = []
    if single_query:
        for song in songs_in_dictionary:
            filter_songs = [int(x[0]) for x in entity_most_similar(model.wv, song, topn=topk, restrict_vocab=search_song_indices) if x[1] >threshold]
            # similar_words = [x[0] for x in model.wv.most_similar(song, topn=topk) if x[1] > threshold]
            # similar_songs = [int(x) for x in similar_words if x.isdigit()]
            # filter_songs = [int(x) for x in similar_songs if x in filtered_song]
            mid_list.extend(filter_songs)
    else:
        filter_songs = [int(x[0]) for x in entity_most_similar(model.wv, songs_in_dictionary, topn=topk, restrict_vocab=search_song_indices) if x[1] > threshold]
        # similar_words = [x[0] for x in model.wv.most_similar(songs_in_dictionary, topn=topk) if x[1] > threshold]
        # similar_songs = [int(x) for x in similar_words if x.isdigit()]
        # filter_songs = [int(x) for x in similar_songs if x in filtered_song]
        mid_list.extend(filter_songs)
    count_candidate_set = Counter(mid_list)
    return count_candidate_set
    
def song_retrieval(songs_in_dictionary,filtered_song, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 100:
        threshold = threshold - 0.1
        topk = topk + 100
        count_candidate_set = _get_candidate_set(songs_in_dictionary,filtered_song, threshold, topk, single_query=1)
        
    result100 = [i[0] for i in count_candidate_set.most_common(100)]
    
    return result100

def _get_tag_candidate_set(tags_in_dictionary,filtered_tag, threshold, topk, single_query=1):
    mid_list = []
    if single_query:
        for tag in tags_in_dictionary:
            filter_tags = [x[0] for x in entity_most_similar(model.wv, tag, topn=topk, restrict_vocab=search_tag_indices)  if x[1] > threshold]
            # similar_words = [x[0] for x in model.wv.most_similar(tag, topn=topk) if x[1] > threshold]
            # similar_tags = [x for x in similar_words if 'p' not in x and not x.isdigit()]
            # filter_tags = [x for x in similar_tags if x in filtered_tag]
            mid_list.extend(filter_tags)
    else:
        filter_tags = [x[0] for x in entity_most_similar(model.wv, tags_in_dictionary, topn=topk, restrict_vocab=search_tag_indices)  if x[1] > threshold]
        # similar_words = [x[0] for x in model.wv.most_similar(tags_in_dictionary, topn=topk) if x[1] > threshold]
        # similar_tags = [x for x in similar_words if 'p' not in x and not x.isdigit()]
        # filter_tags = [x for x in similar_tags if x in filtered_tag]
        mid_list.extend(filter_tags)

    count_candidate_set = Counter(mid_list)
    return count_candidate_set

def tag_retrieval(tags_in_dictionary,filtered_tag, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 10:
        threshold = threshold - 0.01
        topk = topk + 100
        count_candidate_set = _get_tag_candidate_set(tags_in_dictionary,filtered_tag, threshold, topk, single_query=1)
        # if topk > 100:
        #     count_candidate_set = Counter(["기분전환", "감성", "휴식", "발라드", "잔잔한", "드라이브", "힐링", "사랑", "새벽", "밤"])
        #     break
            
    result10 = [i[0] for i in count_candidate_set.most_common(10)]
    
    return result10

def tag_to_song_retrieval(tags_in_dictionary,filtered_song, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 100:
        threshold = threshold - 0.1
        topk = topk + 100
        count_candidate_set = _get_candidate_set(tags_in_dictionary,filtered_song, threshold, topk)
    result100 = [i[0] for i in count_candidate_set.most_common(100)]
    
    return result100

def song_to_tag_retrieval(songs_in_dictionary,filtered_tag, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 10:
        threshold = threshold - 0.01
        topk = topk + 100
        count_candidate_set = _get_tag_candidate_set(songs_in_dictionary,filtered_tag, threshold, topk)
        # if topk > 100:
        #     count_candidate_set = Counter(["기분전환", "감성", "휴식", "발라드", "잔잔한", "드라이브", "힐링", "사랑", "새벽", "밤"])
        #     break
    result10 = [i[0] for i in count_candidate_set.most_common(10)]
    
    return result10


def title_tokenizer(title):
    token = api.analyze(title)
    sentence = []
    try:
        for i in token:
            if i.morphs[0].tag in ['NNG', 'NNP', 'VA', 'SL', 'XR', 'MAG'] and len(i.morphs[0].lex) > 1:
                sentence.append(i.morphs[0].lex)
    except:
        pass
    
    return sentence