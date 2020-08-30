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

api = KhaiiiApi()

with open(os.path.join('./data/src','unique_dict.json')) as f:
    unique_dict = json.load(f)

with open(os.path.join('./data/src','filterSet')) as f:
    filterSet = json.load(f)
    
filtered_tag = filterSet['tagsCum80']
filtered_song = filterSet['freq_cum90Filter']['songs']


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
            # tags_in_dictionary = [str(x) for x in q['tags'] if str(x) in model.wv.vocab]
            tags_in_dictionary = [y for x in q['tags'] if x in unique_dict for y in unique_dict[x] if y in model.wv.vocab] 
            # tags_in_dictionary = [x for x in tags_in_dictionary if x in model.wv.vocab]
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

    write_json(recommends, "results/" +"single_" + types + model_name + ".json")


if __name__ == '__main__':
    val = load_json('./data/val.json')
    test = load_json('./data/test.json')
    model_name = "1003001050sg_model"
    model = gensim.models.Word2Vec.load("./data/models/" + model_name)
    search_song_indices = [model.wv.vocab[str(x)].index for x in filtered_song if str(x) in model.wv.vocab]
    search_tag_indices = list(set([model.wv.vocab[x].index for x in filtered_tag if x in model.wv.vocab]))
    main(model, model_name, "test", test, search_song_indices, search_tag_indices)
