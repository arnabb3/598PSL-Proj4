import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(layout="wide", page_title="Movie Recommender")

# Functions from Part 2/System 2


# Get data ratings training data, and similarity matrix returned from part 1
@st.cache_data
def load_data():
    # Load ratings and movies data
    ratings = pd.read_csv('ratings.dat', sep='::', engine='python', header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movies = pd.read_csv('movies.dat', sep='::', engine='python',
                        encoding="ISO-8859-1", header=None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    
    # Load pre-computed similarity matrices
    # sm = pd.read_csv('sm.csv', index_col=0)
    sm_30 = pd.read_csv('sm_30.csv', index_col=0)
    
    return ratings, movies, sm_30

# Get defualt popular movies based on system 1
def pop_movies(ratings, movies):
    movie_pop = ratings.groupby('MovieID').size().reset_index(name='reviews_count')
    movie_pop = movie_pop.merge(movies, on='MovieID')
    movies_pop_list = movie_pop.sort_values(by='reviews_count', ascending=False).head(100)
    movies_pop_list['image_path'] = movies_pop_list['MovieID'].apply(lambda x: 'MovieImages/' + str(x) + '.jpg')
    return movies_pop_list[['MovieID', 'Title', 'reviews_count', 'image_path']]

#Finds the top 10 recommendations
def myIBCF(newuser, similarity_matrix, popularity_rankings=None):
    # Ensure user ratings have the same index as similarity matrix
    newuser = pd.Series(newuser, index=similarity_matrix.columns)
    rated_movies = newuser.dropna().index
    preds = {}

    # Find all simlar movies to rated movies only
    for movie in similarity_matrix.index:
        if movie not in rated_movies:
            sim_movies = similarity_matrix.loc[movie][rated_movies].dropna()
            if len(sim_movies) > 0 and sim_movies.sum() != 0:
                # Convert to float to avoid Series comparison issues
                sum_ratings = float((newuser[sim_movies.index] * sim_movies).sum())
                mag_sims = float(sim_movies.abs().sum())
                if mag_sims != 0:
                    preds[movie] = sum_ratings / mag_sims

    # If number of predictions greater than 10
    if len(preds) >= 10:
        recs = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:10]
        return [movie for movie, _ in recs]
    
    # If number of predictions < 10 
    else:
        # Fetch defaullt rankings
        if popularity_rankings is None:
            popularity_rankings = [f"m{x}" for x in pop_movies(ratings, movies)['MovieID'].values]
        
        # Sort movies by ratings
        top_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        recs = [movie for movie, _ in top_preds]
        
        num_left = 10 - len(recs)
        for movie in popularity_rankings:
            if num_left == 0:
                break
            if movie not in recs and movie not in rated_movies:
                recs.append(movie)
                num_left -= 1
                
        return recs

# Load data
ratings, movies, sm_30 = load_data()
# only returns top 100 movies for latency/lag purposes
popular_movies = pop_movies(ratings, movies)

# Session state initialization
# To keep track of ratings user has selected
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = pd.Series(np.nan, index=sm_30.columns)

# Title
st.title("🎬 CS 598 PSL Project 4")

# Search and Rating Section - Select up to 10 ratings here
with st.expander("Rate Movies (Maximum 10)", expanded=True):
    
    # Search bar
    search_bar = st.text_input("🔍 Search for movies", "")
    
    # Filter movies based on search
    filtered_movies = popular_movies
    if search_bar:
        filtered_movies = popular_movies[
            popular_movies['Title'].str.contains(search_bar, case=False)
        ]
    
    # Display movies in a grid with ratings
    # 4 movies in a row
    cols = st.columns(4)
    num_rated = sum(st.session_state.user_ratings.notna())
    

    for idx, movie in filtered_movies.iterrows():
        # % 4 to figure out when to go to next row of images 
        col = cols[idx % 4]
        with col:
            try:
                if Path(movie['image_path']).exists():
                    st.image(movie['image_path'], width=150)
                else:
                    st.write("No image available")
            except:
                st.write("No image available")
            
            # Display title and rating widget
            st.write(f"**{movie['Title']}**")
            movie_id = f"m{movie['MovieID']}"
            
            current_rating = st.session_state.user_ratings.get(movie_id, 0)
            can_rate = num_rated < 10 or current_rating > 0
            
            # adds in stars for rating
            selected = st.feedback("stars", key=f"rating_{movie_id}")
            if selected is not None:
                st.session_state.user_ratings[movie_id] = selected + 1
            elif movie_id in st.session_state.user_ratings.index and selected is None:
                st.session_state.user_ratings[movie_id] = np.nan

# Recommendations Section
st.header("📺 Top 10 Recommended Movies")

if st.session_state.user_ratings.notna().sum() == 0:
    # Show popular movies if no ratings
    recommendations = popular_movies.head(10)
else:
    # Get recommendations using pre-computed similarity matrix
    rec_ids = myIBCF(st.session_state.user_ratings, sm_30)
    
    # Create recommendations DataFrame
    recommendations = []
    for movie_id in rec_ids:
        id_num = int(movie_id[1:])  # Remove 'm' prefix and convert to int
        movie_info = movies[movies['MovieID'] == id_num].copy()
        if not movie_info.empty:
            movie_info['image_path'] = f'MovieImages/{id_num}.jpg'
            recommendations.append(movie_info)
    
    recommendations = pd.concat(recommendations)

# Create and display table with images (only shows top 10)
st.write("Based on your ratings, we recommend these movies:")

# Create two columns in the table
cols = st.columns([3, 1])

# Headers
cols[0].write("### Movie Title")
cols[1].write("### Movie Poster")

# Add a separator
st.markdown("---")

# Display each movie in the table
for idx, movie in recommendations.iterrows():
    cols = st.columns([3, 1])
    
    # Movie title column
    cols[0].write(f"**{movie['Title']}**")
    
    # Movie poster column
    try:
        if Path(movie['image_path']).exists():
            cols[1].image(movie['image_path'], width=100)
        else:
            cols[1].write("No image available")
    except:
        cols[1].write("No image available")
    
    # Add a separator between movies
    st.markdown("---")

# Display current ratings selected by user in sidebar
if st.session_state.user_ratings.notna().sum() > 0:
    st.sidebar.header("Your Current Ratings")
    rated_movies = st.session_state.user_ratings[st.session_state.user_ratings.notna()]
    for movie_id, rating in rated_movies.items():
        movie_title = movies.loc[movies['MovieID'] == int(movie_id[1:]), 'Title'].iloc[0]
        st.sidebar.write(f"{movie_title}: {'⭐' * int(rating)}")