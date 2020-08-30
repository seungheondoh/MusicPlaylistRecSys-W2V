import pickle
import json
import os
import re
import random
import numpy as np
from numpy.linalg import norm
import gensim
import multiprocessing

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from khaiii import KhaiiiApi
api = KhaiiiApi()

flatten = lambda l: [item for sublist in l for item in sublist]

def make_unique_dict(tag_list):
    flatten_tags = flatten(tag_list)
    tag_counter = Counter(flatten_tags).most_common()
    df_tag_counter = pd.DataFrame(tag_counter, columns=['tags','freq'])
    delete_list = ['노래','음악','플레이리스트']
    retreval_tags = []
    merge_list = []
    for i in df_tag_counter['tags'][:673]:
        if i in delete_list:
            pass
        else:
            retreval_tags.append(i)
            temp = []
            temp.append(i)
            for j in df_tag_counter['tags'][673:]:
                if i.lower() in j.lower():
                    temp.append(j)
            merge_list.append(temp)

    unique_dict = {}
    for i in merge_list:
        for jdx, j in enumerate(i):
            if j in unique_dict:
                unique_dict[j].append(i[0])
            else:
                unique_dict[j] = [i[0]]
    return unique_dict

def load_dataset(dataset_path):
    # 장르 전체
    genre_gn_all = pd.read_json(os.path.join(dataset_path,'genre_gn_all.json'), typ = 'series')
    genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})

    # 대분류 장르
    gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']

    # 상세 장르 코드
    dtl_gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] != '00']
    dtl_gnr_code.rename(columns = {'gnr_code' : 'dtl_gnr_code', 'gnr_name' : 'dtl_gnr_name'}, inplace = True)

    # 장르 코드 트리
    gnr_code = gnr_code.assign(join_code = gnr_code['gnr_code'].str[0:4])
    dtl_gnr_code = dtl_gnr_code.assign(join_code = dtl_gnr_code['dtl_gnr_code'].str[0:4])

    # Dictionary로 전환
    gnr_dict = gnr_code[['gnr_code','gnr_name']].set_index('gnr_code').to_dict()['gnr_name']
    dtl_gnr_dict= dtl_gnr_code[['dtl_gnr_code','dtl_gnr_name']].set_index('dtl_gnr_code').to_dict()['dtl_gnr_name']

    train = pd.read_json(os.path.join(dataset_path,'train_custom'), typ = 'frame')
    unique_dict = make_unique_dict(train['tags'])

    with open(os.path.join('./data/src','unique_dict.json'), 'w') as f:
        json.dump(unique_dict, f)
        print('unique dict saved')

    with open('./data/src/ply_song_meta.pkl','rb') as file:
        ply_song_list = pickle.load(file)

    return train, ply_song_list, unique_dict, gnr_dict, dtl_gnr_dict

def custom_dict(sent, newdict):
    for key, val in newdict.items():
        if key in sent:
            sent = sent.replace(key, val)
    return sent

def cleanText(sent):
    result = re.sub('[-=+,#/\?:;^$@*\"※~%ㆍ!』\\‘|\(\)\[\]\<\>`\'》]', '', sent)
    result_with_space = re.sub('[.…]', '', result)
    return result_with_space.lower()

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

def title_preprocessing(ply_title_list, unique_dict):
    normalized_title = []
    for title in tqdm(ply_title_list):
        sentence = []
        clean_sentence = cleanText(title)
        try:
            tokenizing = title_tokenizer(replace_sentence)
        except:
            tokenizing = []
        split = clean_sentence.split()
        tokenize_title = list(set(tokenizing + split))
        normalized_title.append(tokenize_title)
        
    return normalized_title

def tag_preprocessing(tag_list):
    normalize_tag = []
    for tags in tqdm(tag_list):
        song_tag = []
        for tag in tags:
            try:
                song_tag += unique_dict[tag]
            except:
                song_tag.append(tag)
        normalize_tag.append(song_tag)
    return normalize_tag

def gnr_to_str(gnr_list,gnr_dict_type):
    nomalize_gnr = []
    for songs_gnr in tqdm(gnr_list):
        temp = []
        for gnr in songs_gnr:
            try:
                temp.append(gnr_dict_type[gnr])
            except:
                temp.append(gnr)
        nomalize_gnr.append(temp)
    
    return nomalize_gnr


def main(dataset_path, save_path, use_meta=0):
    train, ply_song_list, unique_dict, gnr_dict, dtl_gnr_dict = load_dataset(dataset_path)
    df_ply_song = pd.DataFrame(ply_song_list)
    merge_data = pd.merge(train,df_ply_song, on='id')

    normalized_title = title_preprocessing(merge_data['plylst_title'], unique_dict)
    merge_data['normalized_title'] = normalized_title

    normalize_tag = tag_preprocessing(merge_data['tags'])
    merge_data['normalize_tag'] = normalize_tag

    song_ply_gn_gnr_basket_str = gnr_to_str(merge_data['song_ply_gn_gnr_basket'],gnr_dict)
    merge_data['song_ply_gn_gnr_basket_str'] = song_ply_gn_gnr_basket_str

    song_ply_gn_dtl_basket_str = gnr_to_str(merge_data['song_ply_gn_dtl_basket'],dtl_gnr_dict)
    merge_data['song_ply_gn_dtl_basket_str'] = song_ply_gn_dtl_basket_str

    ply_id = []
    for i in merge_data['id']:
        ply_id.append(["p" + str(i)])
    merge_data['ply_id'] = ply_id

    song_str = [[str(w) for w in line] for line in merge_data['songs']]
    song_ply_artist_str = [[str(w) for w in line] for line in merge_data['song_ply_artist']]

    merge_data['song_ply_artist_str'] = song_ply_artist_str
    merge_data['song_str'] = song_str
    train_token = merge_data['normalize_tag'] + merge_data['song_str'] + merge_data['song_ply_gn_gnr_basket_str'] + merge_data['song_ply_gn_dtl_basket_str'] + merge_data['song_ply_artist_str'] + merge_data['normalized_title'] + merge_data['ply_id']
    shuffle_sentences = [ random.sample(sublist, len(sublist)) for sublist in list(train_token) ]

    # Parameter Tunning 필요!
    model = Word2Vec(shuffle_sentences, size=size, window=window, min_count=min_count, iter=iteration , workers=multiprocessing.cpu_count(), sg=1, ns_exponent=-0.5)
    model.save(os.path.join(save_path, str(window) + str(size)+ str(min_count) + str(iteration)+ "sg_model"))
if __name__ == '__main__':
    dataset_path = './data'
    save_path = "./data/models/musical_embedding/model"
    main(dataset_path, save_path)



