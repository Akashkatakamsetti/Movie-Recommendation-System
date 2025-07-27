import pandas as pd
from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

model_path = 'lightgcn_model.pth'

def get_movie_id(movie_name):
    # Function to get the movie ID from its name
    movieid_title = pd.read_csv('dataset\movielensdata.csv')
    for movie_id, title in movieid_title.items():
        if title == movie_name:
            return movie_id
    return None

def recommend_similar_movies(movie_name, num_recs=10):
    movie_id = get_movie_id(movie_name)
    model=movie_id
    if movie_id is None:
        print(f"Movie '{movie_name}' not found.")
        return

    # Get the embeddings for the movie
    e_m = model.items_emb.weight[movie_id]

    # Compute similarity scores between the input movie and all other movies
    scores = torch.matmul(model.items_emb.weight, e_m)

    # Get top recommendations
    values, indices = torch.topk(scores, k=num_recs+1)  # +1 to exclude the input movie itself
    recommended_movies = [(index.cpu().item(), value.item()) for index, value in zip(indices, values) if index != movie_id]

    for index, _ in recommended_movies[:num_recs]:
        movieid_title = get_movie_id(movie_name)
        movie_title = movieid_title.get(index)
        if movie_title:
            print(f"Title: {movie_title}")


def create_similarity():
    data = pd.read_csv('dataset\movielensdata.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data,similarity
def get_suggestions():
    data = pd.read_csv('dataset\movielensdata.csv')
    return list(data['movie_title'].str.capitalize())

def recommend(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

app = Flask(__name__)
@app.route("/")
def index():
    movies=get_suggestions()
    return render_template("index.html",movies=movies)
@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = recommend(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str
if __name__ == '__main__':
    app.run(debug=False)
