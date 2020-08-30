# MelonRec-W2V

- **Team ThunderCat**: ([Public LeaderBoard](https://arena.kakao.com/c/7/leaderboard) 58th)
  
## 1. Competition Overview
- **Goal**: To predict songs and tags that were not given when half or all of the songs and tags in the playlist are **unseen**
- **data**
    - Playlist metadata (title, song, tag, number of likes, update time)
    - Song metadata (song title, album title, genre, title date)
    - Mel-spectrogram of the song
- **Number of participating teams**: 786 teams
  
## 2. Issue
- Challenge: all of the songs and tags in the playlist are **unseen**
- Approach
    - Multi-Modal Retrieval
    - Query by Song, Tag, Title

- **Main Issue**
1. Make co-embedding space that contain song vector, tag vector, title vector
2. Evaluate that the embedding space learns the **semantic relationship**.
3. Cover all retrieval scenario
    - Given Song to Song retrieval (Playlist Continuous)
    - Given Playlist Tag to Tag retrieval (Playlist Auto-tagging)
    - Given Playlist tag to Song retrieval (Unseen Item retrieval)
    - Title to Tag and Song retrieval (Sentence to Item retrieval)

## 3. Contribution

- Approach
    - Make co-embedding space with Word2vec Method
    - Single Modal Retrieval
        - Voting each modality
    - Multi Modal Retrieval
        - Mean each modality
    - Cluster Based Retrieval
    
1. Train Co-embedding Space (Word2Vec Embedding)
    - Input: Sentence (Title token, Tag, Genre, Song, Plylist id)
    - Ouput: Word, Item, Song, Plylist Vector

2. Multi-Modality Retrieval (Inference)

3. Evaluation with ndcg
    - tag wise, song wise


## Dependencies
- numpy 1.16.2
- pandas 0.24.2
- matplotlib 3.0.3
- tqdm 4.31.1
- gensim 3.8.3
- sentencepiece 0.1.91
- sklearn 0.20.3
- khaiii
- pytorch 1.5.1

## 4. Folder and Files
- data download ([link](https://arena.kakao.com/c/7/data))
  - `train.json`, `val.json`, `test.json`, `genre_gn_all.json`, `song_meta.json`


### Learning code (from scratch)
```
$ train_embedding.py
$ embedding_most_similar.py
```   

## 5. To-do
Measure Playlist-Song's Mean and Playlist's Vector Similarity
    - Mid-Evaluation of Embedding Space: KL Divergence
    - Training Method: Self-Supervised Approach

## 6. Reference
- [Musical Word Embedding: Bridging the Gap between Listening Contexts and Music](https://arxiv.org/pdf/2008.01190.pdf)
    - Seungheon Doh, Jongpil Lee, Tae Hong Park, and Juhan Nam. Machine Learning for Media Discovery Workshop, International Conference on Machine Learning (ICML), 2020

- [Automatic music playlist continuation via neighbor-based collaborative filtering and discriminative reweighting/reranking](https://github.com/LauraBowenHe/Recsys-Spotify-2018-challenge).
    - Zhu, L., He, B., Ji, M., Ju, C., & Chen, Y. (2018). In Proceedings of the ACM Recommender Systems Challenge 2018 (pp. 1-6).
