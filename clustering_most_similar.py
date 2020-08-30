import os
import json
import gensim
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from khaiii import KhaiiiApi

from retrieval_utils import *

from sklearn.cluster import KMeans

def get_Kmeans_centroid(model, songs_in_dictionary):
    kmeans = KMeans(n_clusters=2, n_init=5).fit([model.wv[song] for song in songs_in_dictionary])
    return kmeans

with open(os.path.join('../media/bach3/dataset/melonPlaylist/src','unique_dict.json')) as f:
    unique_dict = json.load(f)

with open(os.path.join('../media/bach3/dataset/melonPlaylist/src','filterSet')) as f:
    filterSet = json.load(f)
    
filtered_tag = filterSet['tagsCum80']
filtered_song = filterSet['freq_cum90Filter']['songs']


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


def cleanText(sent):
    result = re.sub('[-=+,#/\?:;^$@*\"※~%ㆍ!』\\‘|\(\)\[\]\<\>`\'》]', '', sent)
    result_with_space = re.sub('[.…]', '', result)
    return result_with_space.lower()

def _get_candidate_set(songs_in_dictionary,filtered_song, threshold, topk, single_query=0):
    mid_list = []
    
    if len(songs_in_dictionary) > 1:
        Kmeans = get_Kmeans_centroid(model, songs_in_dictionary)
        centroid = Kmeans.cluster_centers_
        centroid_label = Kmeans.labels_

        for centroid_vector in centroid:
            filter_songs = [int(x[0]) for x in vector_most_similar(model.wv, centroid_vector, songs_in_dictionary, topn=topk, restrict_vocab=search_song_indices) if x[1] >threshold]
            mid_list.extend(filter_songs)
            
    else:
        for song in songs_in_dictionary:
            filter_songs = [x[0] for x in custom_most_similar(model.wv, song, topn=topk, restrict_vocab=search_tag_indices)  if x[1] > threshold]
            mid_list.extend(filter_songs)
            
    count_candidate_set = Counter(mid_list)
    
    return count_candidate_set
    
def song_retrieval(songs_in_dictionary,filtered_song, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 100:
        threshold = threshold - 0.1
        topk = topk + 100
        count_candidate_set = _get_candidate_set(songs_in_dictionary,filtered_song, threshold, topk, single_query=0)
        
    result100 = [i[0] for i in count_candidate_set.most_common(100)]
    
    return result100

def _get_tag_candidate_set(tags_in_dictionary,filtered_tag, threshold, topk, single_query=0):
    mid_list = []
    
    if len(tags_in_dictionary) > 1:
        Kmeans = get_Kmeans_centroid(model, tags_in_dictionary)
        centroid_tag = Kmeans.cluster_centers_
        centroid_label = Kmeans.labels_

        for centroid_tag_vector in centroid_tag:
            filter_tags = [x[0] for x in vector_most_similar(model.wv, centroid_tag_vector, tags_in_dictionary, topn=topk, restrict_vocab=search_song_indices)  if x[1] > threshold]
            mid_list.extend(filter_tags)
    
    else:
        for tag in tags_in_dictionary:
            filter_tags = [x[0] for x in custom_most_similar(model.wv, tag, topn=topk, restrict_vocab=search_tag_indices)  if x[1] > threshold]
            mid_list.extend(filter_tags)

    count_candidate_set = Counter(mid_list)
    return count_candidate_set

def tag_retrieval(tags_in_dictionary,filtered_tag, threshold, topk):
    count_candidate_set = []
    while len(count_candidate_set) < 10:
        threshold = threshold - 0.01
        topk = topk + 100
        count_candidate_set = _get_tag_candidate_set(tags_in_dictionary,filtered_tag, threshold, topk, single_query=0)
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


def main(model, model_name, types, questions, search_song_indices, search_tag_indices, answers=[], index=0):
    recommends = []
    if index:
        questions = questions[:index]
        if answers != []:
            answers = answers[:index]

    for q in tqdm(questions):
        song_list = []
        tag_list = []
        prior_list = []
        songs_in_dictionary = [str(x) for x in q['songs'] if str(x) in model.wv.vocab]
        
        if q['tags']:
            tags_in_dictionary = [y for x in q['tags'] if x in unique_dict for y in unique_dict[x] if y in model.wv.vocab] 
        else:
            tags_in_dictionary = []
    
        if q['plylst_title']:
            clean_title = cleanText(q['plylst_title']).strip()
            if clean_title == "":
                title_in_dictionary = []
            else:
                tokenizing = title_tokenizer(clean_title)
                split = clean_title.split()
                tokenize_title = list(set(tokenizing + split))
                title_in_dictionary = [str(x) for x in tokenize_title if str(x) in model.wv.vocab]
        else:
            title_in_dictionary = []
        # add title to tags
        tags_in_dictionary += title_in_dictionary
        for title_token in title_in_dictionary:
            if title_token in filtered_tag:
                prior_list.append(title_token)
        if len(songs_in_dictionary) == 0 and len(tags_in_dictionary) == 0 and len(title_in_dictionary) == 0:
            # plylist -> tag, song retrieval
            song_list = MOST_POPULAR
            tag_list = ["기분전환", "감성", "휴식", "발라드", "잔잔한", "드라이브", "힐링", "사랑", "새벽", "밤"]
        
        elif len(songs_in_dictionary) == 0 and len(tags_in_dictionary) == 0:
            song_list = tag_to_song_retrieval(title_in_dictionary,filtered_song, 0.7, 50)
            title_tag_list = tag_retrieval(title_in_dictionary,filtered_tag, 0.3, 20)
            total_tag_list = prior_list + title_tag_list
            tag_list = total_tag_list[:10]
            
        elif len(songs_in_dictionary) == 0:
            song_list = tag_to_song_retrieval(tags_in_dictionary,filtered_song, 0.7, 50)
            tag_list = tag_retrieval(tags_in_dictionary,filtered_tag, 0.3, 20)
            total_tag_list = prior_list + tag_list
            tag_list = total_tag_list[:10]
            
        elif len(tags_in_dictionary) == 0:
            song_list = song_retrieval(songs_in_dictionary,filtered_song, 0.7, 50)
            tag_list = song_to_tag_retrieval(songs_in_dictionary,filtered_tag, 0.7, 50)
            
        else:
            song_list = song_retrieval(songs_in_dictionary,filtered_song, 0.7, 50)
            tag_list = tag_retrieval(tags_in_dictionary,filtered_tag, 0.3, 20)
            total_tag_list = prior_list + tag_list
            tag_list = total_tag_list[:10]
        
        recommends.append({
            "id": q["id"],
            "songs": song_list, 
            "tags":  tag_list,
        })

    write_json(recommends, "results/cluster" + types + model_name + ".json")



if __name__ == '__main__':
    val = load_json('../media/bach3/dataset/melonPlaylist/val.json')
    test = load_json('../media/bach3/dataset/melonPlaylist/test.json')
    model_name = "1003001050sg_model"
    model = gensim.models.Word2Vec.load("../media/bach3/dataset/melonPlaylist/models/" + model_name)
    search_song_indices = [model.wv.vocab[str(x)].index for x in filtered_song if str(x) in model.wv.vocab]
    search_tag_indices = list(set([model.wv.vocab[x].index for x in filtered_tag if x in model.wv.vocab]))
    main(model, model_name, "val", val, search_song_indices, search_tag_indices)
    main(model, model_name, "test", test, search_song_indices, search_tag_indices)


    # model_dir = Path('/media/bach3/dataset/melonPlaylist/models')
    # model_lists = list(model_dir.glob('*50sg_model'))
    # good_models = ['1003001050sg_model' ]
    # for model_path in model_lists:
    #     print(str(model_path))

    #     model = gensim.models.Word2Vec.load(str(model_path))
    #     search_song_indices = [model.wv.vocab[str(x)].index for x in filtered_song if str(x) in model.wv.vocab]
    #     search_tag_indices = list(set([model.wv.vocab[x].index for x in filtered_tag if x in model.wv.vocab]))
    #     main(model, questions, search_song_indices, search_tag_indices, answers=answers, index=500)



def mix_two_lists(l1, l2):
    if len(l1) > (l2):
        long_list = l1
        short_list = l2
    else:
        longer_list = l2
        short_list = l1

    length_ratio = long_list / short_list
