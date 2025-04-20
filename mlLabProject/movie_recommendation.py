import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import requests
from io import BytesIO
from PIL import Image


df = pd.read_csv("mlLabProject/movies.csv")


def recommend_movies(movie_name, df):
    selected = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected:
        df[feature] = df[feature].fillna('')
    
    combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vector)
    
    list_titles = df['title'].tolist()
    find_close = difflib.get_close_matches(movie_name, list_titles)
    if not find_close:
        return ["No similar movies found."]
    
    close_match = find_close[0]
    movie_index = df[df.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[movie_index]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        if i < 15:
            title = df[df.index == index]['title'].values[0]
            genre = df[df.index == index]['genres'].values[0]
            tagline = df[df.index == index]['tagline'].values[0]
            cast = df[df.index == index]['cast'].values[0]
            director = df[df.index == index]['director'].values[0]
            recommendations.append((title, genre, tagline, cast, director))
    return recommendations


def fetch_poster(title):
    try:
        response = requests.get(f"https://www.omdbapi.com/?t={title}&apikey=abe1d7d7")
        data = response.json()
        poster_url = data.get('Poster', '')
        if poster_url != 'N/A':
            return Image.open(BytesIO(requests.get(poster_url).content))
        else:
            return None
    except:
        return None



# Streamlit app UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a favorite movie, and discover similar recommendations!")

movie_name = st.text_input("Enter a movie title:", "")


if st.button("Get Recommendations"):
    if movie_name:
        recommendations = recommend_movies(movie_name, df)
        if isinstance(recommendations, list) and all(isinstance(rec, tuple) for rec in recommendations):
            st.subheader("Recommended Movies:")
            for title, genre, tagline, cast, director in recommendations:
                
                st.markdown(
                    f"""
                    <div style="border: 1px solid #444; border-radius: 8px; padding: 16px; margin: 10px 0; background-color: #2d2d2d;">
                        <div style="display: flex; align-items: center;">
                            <div style="margin-right: 16px;">
                    """,
                    unsafe_allow_html=True,
                )
                
                poster_image = fetch_poster(title)
                if poster_image:
                    st.image(poster_image, width=150, caption=title)
                else:
                    st.write(f"**{title}**")

                st.markdown(
                    f"""
                            </div>
                            <div>
                                <h3 style="color: #e5e5e5; margin: 0;">{title}</h3>
                                <p style="color: #cccccc;"><strong>Genre:</strong> {genre}</p>
                                <p style="color: #cccccc;"><strong>Tagline:</strong> {tagline}</p>
                                <p style="color: #cccccc;"><strong>Director:</strong> {director}</p>
                                <p style="color: #cccccc;"><strong>Cast:</strong> {cast.join(',')}</p>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(recommendations[0])
    else:
        st.write("Please enter a movie title.")
