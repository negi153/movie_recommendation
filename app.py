from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

app = Flask(__name__)

def create_movie_recommendation_system():
    # read data
    movies = pd.read_csv(r'dataset/top10K-TMDB-movies.csv')

    # Feature selection : Select some columns which will help in createing Maching learning model
    movies = movies[['id', 'title', 'genre', 'overview']]

    # combine overview and genre column
    movies['tags'] = movies['title'] + movies['overview'] + movies['genre']  

    # delete unnecessary columns
    movies_new_data = movies.drop(columns=['overview', 'genre'])

    # create vector
    cv = CountVectorizer(max_features=10000, stop_words='english')
    vector = cv.fit_transform(movies_new_data['tags'].values.astype('U')).toarray() # converting tags column into UTF and then to vector

    # create cosine similarity
    similarity = cosine_similarity(vector)

    return movies_new_data, similarity

movies_data, similarity = create_movie_recommendation_system()

# # Reading data from pickle file
# movies_data = pickle.load(open(r'NLP_model/movies.pkl','rb'))
# similarity = pickle.load(open(r'NLP_model/movies_similarity.pkl','rb'))

def fetch_movie_poster(movie_id):
    '''
        This method gets the poster of movies
    '''
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=2bf71125b4aeeff9bdc25e091efcb1f7"
    data = requests.get(url)

    data = data.json()
    poster_path = data['poster_path']

    return "https://image.tmdb.org/t/p/w500" + poster_path


# Dummy movie recommendation function
def get_movie_recommendations(movie_name):
    
    # find the index of movie in dataframe
    index = movies_data[movies_data['title'] == movie_name].index[0] 
    
    # sort the similarity vector in reverse order
    distance = sorted(list(enumerate(similarity[index])), reverse = True, key = lambda vector:vector[1])
    
    recommendations = []
    for i in distance[1:6]:
        # print(movies_data.iloc[i[0]].title)
        
        # movie_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/182px-Python-logo-notext.svg.png'

        movie_url = fetch_movie_poster(movies_data.iloc[i[0]].id)

        recommendations.append((movies_data.iloc[i[0]].title,movie_url))

    return recommendations

@app.route('/')
def index():
    # return render_template('index.html', recommended_movies=[])
    return render_template('index.html', all_movies = movies_data['title'])

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_name = request.form['movie_name']
    recommended_movies = get_movie_recommendations(movie_name)
    return render_template('index.html', all_movies = movies_data['title'], recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)