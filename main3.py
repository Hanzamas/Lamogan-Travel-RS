import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
from tensorflow import keras
from surprise import SVD, Dataset, Reader
import numpy as np

# Buat daftar stop words bahasa Indonesia menggunakan Sastrawi
factory = StopWordRemoverFactory()
indonesian_stop_words = factory.get_stop_words()

# Load datasets
@st.cache_data
def load_data():
    tourism_rating = pd.read_csv("data/tourism_rating.csv", encoding="ascii")
    tourism_with_id = pd.read_csv("data/tourism_with_id.csv", encoding="Johab")
    user_data = pd.read_csv("data/user.csv", encoding="ascii")
    return tourism_rating, tourism_with_id, user_data


tourism_rating, tourism_with_id, user_data = load_data()

# Preprocess Data
def preprocess_data():
    tourism_with_id['Jumlah Ulasan'] = tourism_with_id['Jumlah Ulasan'].str.replace(',', '').astype(int)
    # merged_data = pd.merge(tourism_rating, tourism_with_id, on="Place_Id")
    tourism_with_id['combined'] = tourism_with_id['Place_Name'] + ' ' + tourism_with_id['Category']
    return tourism_with_id
    # # Create 'content' column using relevant fields
    # merged_data['content'] = merged_data[
    #     ['Place_Name', 'Category']
    # ].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)
    # Combine 'Place_Name' and 'Category' into a new 'combined' column
    # return merged_data

merged_data = preprocess_data()



# Compute TF-IDF and Cosine Similarity
@st.cache_data
def compute_similarity(data):
    cv = CountVectorizer(stop_words=indonesian_stop_words)
    cv_matrix = cv.fit_transform(data['combined'])
    cosine_sim = cosine_similarity(cv_matrix)
    # indices = pd.Series(data.index, index=data['Place_Name']).drop_duplicates()
    # return cosine_sim, indices
    return cosine_sim
# cosine_sim, indices = compute_similarity(merged_data)
cosine_sim = compute_similarity(merged_data)


# Content-Based Recommendation
def content_based_recommendation(name, cosine_sim, items, n=5):
    # Ensure case-insensitive matching
    matching_items = items[items['combined'].str.contains(name, case=False, na=False)]

    if matching_items.empty:
        st.warning(f"No places found matching '{name}'. Showing fallback recommendations.")
        # Fallback: Top-rated places
        fallback = items.sort_values(by='Rating', ascending=False)
        return fallback[['Place_Name', 'Category', 'Price', 'Description', 'City']].head(n)

    # Get the index of the first matching item
    idx = matching_items.index[0]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top n most similar items
    top_indices = [i[0] for i in sim_scores[1:n + 1]]  # Exclude the input itself
    recommendations = items.iloc[top_indices][['Place_Name', 'Category', 'Price', 'Description', 'City']]

    if recommendations.empty:
        st.warning(f"No similar places found for '{name}'. Showing fallback recommendations.")
        fallback = items.sort_values(by='Rating', ascending=False)
        return fallback[['Place_Name', 'Category', 'Price', 'Description', 'City']].head(n)

    return recommendations


# # Collaborative Filtering
# def collaborative_filtering(data, user_id, n=5):
#     reader = Reader(rating_scale=(0.5, 5))
#     rating_data = data[['User_Id', 'Place_Id', 'Place_Ratings']]
#     dataset = Dataset.load_from_df(rating_data, reader)
#
#     svd = SVD()
#     trainset = dataset.build_full_trainset()
#     svd.fit(trainset)
#
#     if user_id not in rating_data['User_Id'].unique():
#         st.error(f"User ID {user_id} is not found in the dataset!")
#         return pd.DataFrame()
#
#     visited_places = data[data['User_Id'] == user_id]['Place_Id'].tolist()
#     all_places = data['Place_Id'].unique()
#     unvisited_places = [place for place in all_places if place not in visited_places]
#
#     if not unvisited_places:
#         fallback_recommendations = tourism_with_id.sort_values(by=['Rating', 'Jumlah Ulasan'], ascending=[False, False])
#         return fallback_recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)
#
#     predictions = [(place, svd.predict(user_id, place).est) for place in unvisited_places]
#     predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
#
#     recommended_places = [place[0] for place in predictions]
#     recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_places)][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]
#
#     return recommendations

# Updated Collaborative Filtering Function
def collaborative_filtering_with_model(data, user_id, model, n=5):
    try:
        # Ensure user_id exists in the dataset
        user_ids = data.User_Id.unique().tolist()
        place_ids = data.Place_Id.unique().tolist()
        user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
        place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
        place_encoded_to_place = {i: x for x, i in enumerate(place_ids)}

        if user_id not in user_to_user_encoded:
            st.error(f"User ID {user_id} not found in the dataset!")
            return pd.DataFrame()

        user_encoder = user_to_user_encoded[user_id]
        places_visited_by_user = data[data['User_Id'] == user_id]['Place_Id'].tolist()
        unvisited_places = list(set(place_ids) - set(places_visited_by_user))

        if not unvisited_places:
            # User has visited all places
            st.warning(f"User {user_id} has visited all available places.")
            st.info("Recommending the top places from visited ones based on preferences.")

            # Prepare data for places already visited
            visited_encoded = [[place_to_place_encoded.get(x)] for x in places_visited_by_user]
            user_place_array = np.hstack(([[user_encoder]] * len(visited_encoded), visited_encoded))

            # Predict ratings for visited places
            ratings = model.predict(user_place_array).flatten()

            random_noise = np.random.uniform(-0.05, 0.05, len(ratings))
            adjusted_ratings = ratings + random_noise

            # Get top N recommendations from adjusted ratings
            top_ratings_indices = adjusted_ratings.argsort()[-n:][::-1]
            recommended_place_ids = [
                place_encoded_to_place.get(visited_encoded[x][0]) for x in top_ratings_indices
            ]

            # Fetch and display recommended places
            recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_place_ids)]
            return recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)

        # Standard recommendation flow for unvisited places
        unvisited_encoded = [[place_to_place_encoded.get(x)] for x in unvisited_places]
        user_place_array = np.hstack(([[user_encoder]] * len(unvisited_encoded), unvisited_encoded))

        # Predict ratings for unvisited places
        ratings = model.predict(user_place_array).flatten()

        # Get top N recommendations for unvisited places
        top_ratings_indices = ratings.argsort()[-n:][::-1]
        recommended_place_ids = [
            place_encoded_to_place.get(unvisited_encoded[x][0]) for x in top_ratings_indices
        ]

        # Fetch and display recommended places
        recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_place_ids)]
        return recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)

    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return pd.DataFrame()


# Collaborative Filtering with SVD
def collaborative_filtering_svd(data, user_id, n=5):
    # Prepare the dataset for SVD
    reader = Reader(rating_scale=(0.5, 5))
    filtered_data = data[data['Place_Ratings'] > 0]  # Exclude ratings of 0
    dataset = Dataset.load_from_df(filtered_data[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

    # Train the SVD model
    svd = SVD()
    trainset = dataset.build_full_trainset()
    svd.fit(trainset)

    # Get visited places (rating > 0) for the user
    visited_places = data[(data['User_Id'] == user_id) & (data['Place_Ratings'] > 0)]['Place_Id'].tolist()
    all_places = data['Place_Id'].unique().tolist()

    # Handle cases where the user has visited all places
    if not set(all_places) - set(visited_places):
        st.warning(f"User {user_id} has visited all available places.")
        st.info("Recommending the top places from visited ones based on preferences.")

        visited_encoded = [place for place in visited_places]
        user_place_array = [[user_id, place] for place in visited_encoded]

        # Predict ratings for visited places
        ratings = [svd.predict(user_id, place).est for user_id, place in user_place_array]

        # Get top N recommendations from visited places
        top_ratings_indices = sorted(range(len(ratings)), key=lambda i: ratings[i], reverse=True)[:n]
        recommended_place_ids = [visited_encoded[i] for i in top_ratings_indices]

        # Fetch and display recommended places
        recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_place_ids)]
        return recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)

    # Handle unvisited places
    unvisited_places = list(set(all_places) - set(visited_places))
    predictions = [(place, svd.predict(user_id, place).est) for place in unvisited_places]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    # Get the recommended places
    recommended_place_ids = [place for place, _ in predictions]
    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_place_ids)]
    return recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)


import tensorflow as tf
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_place, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_place = num_place
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.place_embedding = layers.Embedding(
            num_place,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.place_bias = layers.Embedding(num_place, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])

        dot_user_place = tf.tensordot(user_vector, place_vector, 2)
        x = dot_user_place + user_bias + place_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        return {
            "num_users": self.num_users,
            "num_place": self.num_place,
            "embedding_size": self.embedding_size,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Load the pre-trained model
@st.cache_resource
def load_recommendation_model():
    try:
        return load_model(
            "recommender_model.keras",
            custom_objects={"RecommenderNet": RecommenderNet}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_recommendation_model()

# Simple Recommendation
def simple_recommender(data, category=None, min_price=None, max_price=None, min_rating=None, min_reviews=None, n=5):
    if category:
        data = data[data['Category'] == category]  # Filter berdasarkan kategori
    if min_price is not None:
        data = data[data['Price'] >= min_price]
    if max_price is not None:
        data = data[data['Price'] <= max_price]
    if min_rating is not None:
        data = data[data['Rating'] >= min_rating]
    if min_reviews is not None:
        data = data[data['Jumlah Ulasan'] >= min_reviews]

    data = data.sort_values(by='Rating', ascending=False)
    return data[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Jumlah Ulasan']].head(n)

# Display User Table
def display_user_table():
    st.subheader("User Information")
    user_table = user_data[['User_Id', 'Name', 'Gender', 'Age', 'Location']].sort_values(by='User_Id')
    st.dataframe(user_table)

def display_statistics():
    st.title("Statistics and Insights")

    # Basic statistics for tourism_with_id
    st.subheader("Basic Statistics for Tourism Data")
    st.write(tourism_with_id.describe())

    # Distribution of Categories
    st.subheader("Category Distribution")
    category_counts = tourism_with_id['Category'].value_counts()
    st.bar_chart(category_counts)

    # Distribution of Ratings
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    tourism_with_id['Rating'].hist(bins=10, ax=ax, grid=False)
    ax.set_title("Distribution of Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # City-wise Distribution
    st.subheader("City-wise Place Distribution")
    city_counts = tourism_with_id['City'].value_counts()
    st.bar_chart(city_counts)

    # Price vs Rating Scatterplot
    st.subheader("Price vs Rating")
    fig, ax = plt.subplots()
    tourism_with_id.plot.scatter(x='Price', y='Rating', alpha=0.5, ax=ax)
    ax.set_title("Price vs Rating")
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    # User Data Statistics
    st.subheader("User Data Insights")

    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_counts = user_data['Gender'].value_counts()
    st.bar_chart(gender_counts)

    # Location Distribution
    st.subheader("Location Distribution")
    location_counts = user_data['Location'].value_counts()
    st.bar_chart(location_counts)

    # Age Distribution
    st.subheader("Age Distribution")
    user_data['Age_Group'] = pd.cut(user_data['Age'], bins=[0, 20, 30, 40, 100],
                                    labels=["<20", "20-30", "30-40", ">40"])
    age_group_counts = user_data['Age_Group'].value_counts()
    st.bar_chart(age_group_counts)

    # Average Ratings by Gender
    st.subheader("Average Ratings by Gender")
    avg_rating_by_gender = tourism_rating.merge(user_data, on='User_Id').groupby('Gender')['Place_Ratings'].mean()
    st.bar_chart(avg_rating_by_gender)

    # Average Ratings by Age Group
    st.subheader("Average Ratings by Age Group")
    avg_rating_by_age = tourism_rating.merge(user_data, on='User_Id').groupby('Age_Group')['Place_Ratings'].mean()
    st.bar_chart(avg_rating_by_age)


# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Recommendation System", "Statistics", "Top 10 Best Places"])

if page == "Recommendation System":
    st.title("Travel Recommendation System")
    st.sidebar.header("Recommendation Options")
    selected_model = st.sidebar.selectbox(
        "Select Recommendation Model:",
        ["Simple Recommendation", "Content-Based Filtering", "Collaborative Filtering", "Collaborative Filtering SVD"]
    )

    if selected_model == "Simple Recommendation":
        st.subheader("Simple Recommendations")

        # Tambahkan filter kategori dan parameter lainnya
        category_options = tourism_with_id['Category'].unique()
        selected_category = st.selectbox("Select a Category (Optional):", [None] + list(category_options))

        min_price = st.number_input("Minimum Price (Optional):", min_value=0, step=1, value=0)
        max_price = st.number_input("Maximum Price (Optional):", min_value=0, step=1, value=0)
        if max_price == 0:
            max_price = None  # Tidak ada batas atas harga jika tidak diatur

        min_rating = st.slider("Minimum Rating (Optional):", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
        min_reviews = st.number_input("Minimum Number of Reviews (Optional):", min_value=0, step=1, value=0)

        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Get Recommendations"):
            recommendations = simple_recommender(
                tourism_with_id,
                category=selected_category,
                min_price=min_price if min_price > 0 else None,
                max_price=max_price,
                min_rating=min_rating if min_rating > 0 else None,
                min_reviews=min_reviews if min_reviews > 0 else None,
                n=num_recommendations
            )
            st.write("Here are the top recommended places:")
            st.dataframe(recommendations)



    elif selected_model == "Collaborative Filtering":

        st.subheader("Collaborative Recommendations")
        # Show user table
        display_user_table()
        user_id = st.number_input("Enter User ID:", min_value=1, step=1)

        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend Based on User Preferences"):

            if model:

                recommendations = collaborative_filtering_with_model(

                    tourism_rating, user_id, model, n=num_recommendations

                )

                st.write("Here are the recommendations based on your preferences:")

                st.dataframe(recommendations)

            else:

                st.error("Model not loaded!")

    if selected_model == "Content-Based Filtering":

        st.subheader("Content-Based Recommendations")

        selected_place = st.text_input("Enter a place name:")

        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend"):
            recommendations = content_based_recommendation(

                name=selected_place,

                cosine_sim=cosine_sim,

                items=merged_data,

                n=num_recommendations

            )

            st.write("Recommended Places:")

            st.dataframe(recommendations)

    if selected_model == "Collaborative Filtering SVD":
        st.title("Collaborative Filtering with SVD")
        # Show user table
        display_user_table()
        # User input for User ID
        user_id = st.number_input("Enter User ID:", min_value=1, step=1)
        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend Based on Collaborative Filtering"):
            if user_id not in tourism_rating['User_Id'].unique():
                st.error(f"User ID {user_id} is not found in the dataset!")
            else:
                recommendations = collaborative_filtering_svd(tourism_rating, user_id, n=num_recommendations)
                st.write("Here are the recommendations based on your preferences:")
                st.dataframe(recommendations)



elif page == "Statistics":
    display_statistics()

elif page == "Top 10 Best Places":
    st.title("Top 10 Best Places to Visit")

    # Compute the top 10 places based on ratings and the number of reviews
    top_places = tourism_with_id.sort_values(by=['Rating', 'Jumlah Ulasan'], ascending=[False, False]).head(10)

    st.subheader("Here are the top 10 best-rated places by Ratings and Review:")
    st.dataframe(top_places[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Jumlah Ulasan']])