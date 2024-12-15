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
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Buat daftar stop words bahasa Indonesia menggunakan Sastrawi
factory = StopWordRemoverFactory()
indonesian_stop_words = factory.get_stop_words()
base64_logo = image_to_base64("data/logo.jpg")  # Convert the logo to Base64

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

# Preprocess Data
def preprocess_data2():
    # tourism_with_id['Jumlah Ulasan'] = tourism_with_id['Jumlah Ulasan'].str.replace(',', '').astype(int)
    # merged_data = pd.merge(tourism_rating, tourism_with_id, on="Place_Id")
    tourism_with_id['combined'] = tourism_with_id['Place_Name'] + ' ' + tourism_with_id['Category'] + ' ' + tourism_with_id['Description']
    return tourism_with_id
    # # Create 'content' column using relevant fields
    # merged_data['content'] = merged_data[
    #     ['Place_Name', 'Category']
    # ].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)
    # Combine 'Place_Name' and 'Category' into a new 'combined' column
    # return merged_data

merged_data2 = preprocess_data2()

from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_data
def compute_similarity_tfidf(data):
    # Initialize the TF-IDF vectorizer with stop words
    tfidf = TfidfVectorizer(stop_words=indonesian_stop_words)

    # Fit and transform the 'combined' column
    tfidf_matrix = tfidf.fit_transform(data['combined'])

    # Compute cosine similarity from the TF-IDF matrix
    cosine_sim = cosine_similarity(tfidf_matrix)

    return cosine_sim


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
cosine_sim_tfidf = compute_similarity_tfidf(merged_data2)

def content_based_recommendation_tfidf(name, cosine_sim_tfidf, items, n=5):
    # Ensure case-insensitive matching
    matching_items = items[items['combined'].str.contains(name, case=False, na=False)]

    if matching_items.empty:
        st.warning(f"No places found matching '{name}'. Showing fallback recommendations.")
        # Fallback: Top-rated places
        fallback = items.sort_values(by='Rating', ascending=False)
        return fallback[['Place_Name', 'Category', 'Price', 'Description', 'City']].head(n)

    # Get the index of the first matching item
    idx = matching_items.index[0]

    # Compute similarity scores based on TF-IDF
    sim_scores = list(enumerate(cosine_sim_tfidf[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top n most similar items
    top_indices = [i[0] for i in sim_scores[1:n + 1]]  # Exclude the input itself
    recommendations = items.iloc[top_indices][['Place_Name', 'Category', 'Price', 'Description', 'City']]

    if recommendations.empty:
        st.warning(f"No similar places found for '{name}'. Showing fallback recommendations.")
        fallback = items.sort_values(by='Rating', ascending=False)
        return fallback[['Place_Name', 'Category', 'Price', 'Description', 'City']].head(n)

    return recommendations

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
from surprise import KNNBasic
# Item-Based Collaborative Filtering
def item_based_recommendation(data, item_id, tourism_with_id, n=5):
    # Ensure Place_IDs are strings in both datasets and input
    tourism_rating['Place_Id'] = tourism_rating['Place_Id'].astype(str)
    tourism_with_id['Place_Id'] = tourism_with_id['Place_Id'].astype(str)
    item_id = str(item_id)  # Convert input ID to string

    # Ensure input is a string
    item_id = str(item_id)

    # Train the model as before
    reader = Reader(rating_scale=(0.5, 5))
    dataset = Dataset.load_from_df(data[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

    sim_options = {
        'name': 'cosine',
        'user_based': False
    }
    item_based_model = KNNBasic(sim_options=sim_options)
    trainset = dataset.build_full_trainset()
    item_based_model.fit(trainset)

    # Debug: Check if item_id exists in training set
    if item_id not in trainset._raw2inner_id_items:
        st.warning(f"Place ID {item_id} not found in the training dataset.")
        st.write("Internal Item IDs (Surprise):", trainset._raw2inner_id_items)  # Debug info
        return tourism_with_id.sort_values(by='Rating', ascending=False).head(n)

    # Proceed with recommendation
    item_inner_id = trainset.to_inner_iid(item_id)
    neighbors = item_based_model.get_neighbors(item_inner_id, k=n)
    similar_items = [trainset.to_raw_iid(inner_id) for inner_id in neighbors]

    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(similar_items)]
    return recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)


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

        # Map user and places to encoded values
        user_encoder = user_to_user_encoded[user_id]

        # Extract places visited by the user with ratings > 0
        places_visited_by_user = data[(data['User_Id'] == user_id) & (data['Place_Ratings'] > 0)]['Place_Id'].tolist()

        # Extract unvisited places (those with a rating of 0)
        unvisited_places = data[(data['User_Id'] == user_id) & (data['Place_Ratings'] == 0)]['Place_Id'].tolist()

        if not unvisited_places:
            # User has no unrated places
            st.warning(f"User {user_id} has visited and rated all available places.")
            st.info("Recommending the top places from visited ones based on preferences.")

            # Prepare data for places already visited
            visited_encoded = [[place_to_place_encoded.get(x)] for x in places_visited_by_user]
            user_place_array = np.hstack(([[user_encoder]] * len(visited_encoded), visited_encoded))

            # Predict ratings for visited places
            ratings = model.predict(user_place_array).flatten()

            # Add a small random noise to diversify results
            random_noise = np.random.uniform(-0.01, 0.01, len(ratings))
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

def compute_top_places(data, n=10):
    # Calculate the mean rating across all places
    C = data['Rating'].mean()

    # Calculate the 75th percentile of the number of reviews
    m = data['Jumlah Ulasan'].quantile(0.75)

    # Filter places with at least `m` reviews
    qualified = data[data['Jumlah Ulasan'] >= m]

    # Calculate the weighted score
    qualified['score'] = (qualified['Jumlah Ulasan'] / (qualified['Jumlah Ulasan'] + m) * qualified['Rating']) + \
                         (m / (qualified['Jumlah Ulasan'] + m) * C)

    # Sort by score and return the top n places
    top_places = qualified.sort_values('score', ascending=False).head(n)
    return top_places[['Place_Name', 'Category', 'City', 'Rating', 'Jumlah Ulasan', 'score']]

# Display User Table
def display_user_table():
    st.subheader("User Information")
    user_table = user_data[['User_Id', 'Name', 'Gender', 'Age', 'Location']].sort_values(by='User_Id')
    st.dataframe(user_table)
def display_wisata_table():
    st.subheader("Tempat Wisata")
    wisata_table = tourism_with_id[['Place_Id', 'Place_Name', 'Category', 'City', 'Rating', 'Price', 'Jumlah Ulasan']].sort_values(by='Place_Id')
    st.dataframe(wisata_table)

def display_statistics():
    st.header("Statistik Data Tempat Wisata dan Pengguna")
    with st.expander("Penjelasan Statistik"):
        st.write("""
        Pada halaman ini, Anda dapat menemukan berbagai informasi statistik dan wawasan terkait data wisata.
        Statistik dasar seperti distribusi kategori tempat, distribusi rating, jumlah tempat berdasarkan kota, 
        serta hubungan antara harga dan rating disajikan dalam bentuk grafik dan tabel.
        Halaman ini bertujuan untuk membantu pengguna memahami pola data secara keseluruhan sebelum menggunakan sistem rekomendasi.
        """)
    # Basic statistics for tourism_with_id
    st.subheader("Statistik Dasar Data Tempat Wisata")
    st.write(tourism_with_id.describe())

    # Distribution of Categories
    st.subheader("Kategori Distribusi")
    category_counts = tourism_with_id['Category'].value_counts()
    st.bar_chart(category_counts)

    # Distribution of Ratings
    st.subheader("Distribusi Rating")
    fig, ax = plt.subplots()
    tourism_with_id['Rating'].hist(bins=10, ax=ax, grid=False)
    ax.set_title("Distribusi Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # City-wise Distribution
    st.subheader("Distribusi Kota Wisata")
    city_counts = tourism_with_id['City'].value_counts()
    st.bar_chart(city_counts)

    # Price vs Rating Scatterplot
    st.subheader("Harga vs Rating")
    fig, ax = plt.subplots()
    tourism_with_id.plot.scatter(x='Price', y='Rating', alpha=0.5, ax=ax)
    ax.set_title("Harga vs Rating")
    ax.set_xlabel("Harga")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

    # User Data Statistics
    st.subheader("Statistik Pengguna")

    # Gender Distribution
    st.subheader("Distribusi Gender")
    gender_counts = user_data['Gender'].value_counts()
    st.bar_chart(gender_counts)

    # Location Distribution
    st.subheader("Distribusi Lokasi Pengguna")
    location_counts = user_data['Location'].value_counts()
    st.bar_chart(location_counts)

    # Age Distribution
    st.subheader("Distribusi Umur Pengguna")
    user_data['Age_Group'] = pd.cut(user_data['Age'], bins=[0, 20, 30, 40, 100],
                                    labels=["<20", "20-30", "30-40", ">40"])
    age_group_counts = user_data['Age_Group'].value_counts()
    st.bar_chart(age_group_counts)

    # Average Ratings by Gender
    st.subheader("Rata-rata Rating berdasarkan Gender")
    avg_rating_by_gender = tourism_rating.merge(user_data, on='User_Id').groupby('Gender')['Place_Ratings'].mean()
    st.bar_chart(avg_rating_by_gender)

    # Average Ratings by Age Group
    st.subheader("Rata-rata rating berdasarkan Grup Umur")
    avg_rating_by_age = tourism_rating.merge(user_data, on='User_Id').groupby('Age_Group')['Place_Ratings'].mean()
    st.bar_chart(avg_rating_by_age)

def add_footer():
    st.markdown(
        """
        <style>
        footer {
            visibility: hidden;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f0f0;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #555;
            border-top: 1px solid #ddd;
        }
        </style>
        <div class="footer">
            © 2024 Lamongan Travel Recommendation System. All rights reserved. Bisri Copyright.
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
# st.sidebar.title("Navigasi Sistem Rekomendasi Tempat Wisata")
# page = st.sidebar.radio("Go to:", ["Sistem Rekomendasi", "Statistik", "Top 10 tempat terbaik"])
st.set_page_config(layout="wide", page_title="Sistem Rekomendasi Tempat Wisata Lamongan", page_icon="logo.jpg")
# st.markdown("""
# # 🌍 Sistem Rekomendasi Tempat Wisata Lamongan
# ---
# Sistem ini dirancang untuk membantu pengguna menemukan tempat wisata terbaik berdasarkan kebutuhan dan preferensi mereka. Dengan berbagai metode rekomendasi seperti **Content-Based Filtering**, **Collaborative Filtering**, dan **Hybrid Recommendations**, aplikasi ini memberikan pengalaman yang personal dan relevan.
#
# Berikut adalah fitur utama dalam sistem ini:
# - **Sistem Rekomendasi**: Menyediakan berbagai metode rekomendasi.
# - **Statistik**: Menampilkan data dan wawasan dari basis data tempat wisata.
# - **Top 10 Tempat Terbaik**: Memberikan daftar tempat wisata dengan ulasan terbaik berdasarkan analisis data.
#
# Kelompok Data Science Melon - Kelas A
# """)


# Add a logo and title in the header
# Add logo and title
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/jpeg;base64,{base64_logo}" alt="Logo" width="50" style="margin-right: 15px;">
        <h1 style="display: inline; vertical-align: middle;">Sistem Rekomendasi Tempat Wisata Lamongan</h1>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# App description below the title
st.markdown("""
Sistem ini dirancang untuk membantu pengguna menemukan tempat wisata terbaik berdasarkan kebutuhan dan preferensi mereka. Dengan berbagai metode rekomendasi seperti **Content-Based Filtering**, **Collaborative Filtering**, dan **Hybrid Recommendations**, aplikasi ini memberikan pengalaman yang personal dan relevan.

Berikut adalah fitur utama dalam sistem ini:
- **Sistem Rekomendasi**: Menyediakan berbagai metode rekomendasi.
- **Statistik**: Menampilkan data dan wawasan dari basis data tempat wisata.
- **Top 10 Tempat Terbaik**: Memberikan daftar tempat wisata dengan ulasan terbaik berdasarkan analisis data.

Kelompok Data Science Melon - Kelas A
""")

tab1, tab2, tab3 = st.tabs(["Sistem Rekomendasi", "Statistik", "Top 10 Tempat Terbaik"])
# if page == "Sistem Rekomendasi":
with tab1:
    st.title("Rekomendasi Tempat Wisata")
    with st.expander("Apa itu Sistem Rekomendasi Tempat Wisata?"):
        st.write("""
        Sistem rekomendasi tempat wisata adalah sebuah teknologi yang dirancang untuk membantu pengguna menemukan destinasi wisata berdasarkan preferensi, minat, dan kebutuhan mereka. Sistem ini menggunakan data seperti riwayat interaksi pengguna, ulasan, deskripsi tempat, dan informasi lain untuk memberikan rekomendasi yang relevan dan personal. Sistem rekomendasi tempat wisata yang Anda gunakan menggabungkan beberapa pendekatan modern untuk memberikan rekomendasi terbaik.
        """)
    st.header("Opsi Metode Rekomendasi")
    selected_model = st.selectbox(
        "Pilih Metode Rekomendasi:",
        ["Simple Recommendation", "Content-Based Filtering" , "Content-Based Filtering+", "Collaborative Filtering RecommenderNet", "Collaborative Filtering SVD", "Item-Based Collaborative Filtering"]
    )

    if selected_model == "Simple Recommendation":
        st.subheader("Rekomendasi Sederhana")
        with st.expander("Apa itu Rekomendasi Sederhana?"):
            st.write("""
            Metode Simple Recommendation adalah pendekatan yang paling dasar dalam sistem rekomendasi. Metode ini memungkinkan pengguna untuk memfilter tempat berdasarkan kriteria seperti kategori, rentang harga, rating minimum, dan jumlah ulasan minimum. Tempat yang memenuhi filter diurutkan berdasarkan rating dalam urutan menurun. Metode ini cocok untuk pengguna baru atau yang tidak memiliki data personal.
            """)
        # Tambahkan filter kategori dan parameter lainnya
        category_options = tourism_with_id['Category'].unique()
        selected_category = st.selectbox("Pilih Kategori (Optional):", [None] + list(category_options))

        min_price = st.number_input("Harga Minimal (Optional):", min_value=0, step=1, value=0)
        max_price = st.number_input("Harga Maksimal (Optional):", min_value=0, step=1, value=0)
        if max_price == 0:
            max_price = None  # Tidak ada batas atas harga jika tidak diatur

        min_rating = st.slider("Rating Minimal (Optional):", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
        min_reviews = st.number_input("Minimal Jumlah Review (Optional):", min_value=0, step=1, value=0)

        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi"):
            recommendations = simple_recommender(
                tourism_with_id,
                category=selected_category,
                min_price=min_price if min_price > 0 else None,
                max_price=max_price,
                min_rating=min_rating if min_rating > 0 else None,
                min_reviews=min_reviews if min_reviews > 0 else None,
                n=num_recommendations
            )
            st.write("Berikut ini adalah Rekomendasi nya:")
            st.dataframe(recommendations)



    elif selected_model == "Collaborative Filtering RecommenderNet":

        st.subheader("Collaborative Recommendations RecommenderNet")
        with st.expander("Apa itu Collaborative Filtering (RecommenderNet)?"):
            st.write("""
            Collaborative Filtering menggunakan preferensi pengguna lain yang mirip untuk merekomendasikan tempat. Model TensorFlow Keras RecommenderNet mempelajari hubungan laten antara pengguna dan tempat dari interaksi sebelumnya. Metode ini memberikan rekomendasi yang sangat personal, tetapi membutuhkan data historis pengguna yang cukup.
            """)
        # Show user table
        display_user_table()
        user_id = st.number_input("Masukan User ID:", min_value=1, step=1)

        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi"):

            if model:

                recommendations = collaborative_filtering_with_model(

                    tourism_rating, user_id, model, n=num_recommendations

                )

                st.write("Berikut ini adalah Rekomendasi berdasarkan Referensi Pengguna:")

                st.dataframe(recommendations)

            else:

                st.error("Model not loaded!")

    if selected_model == "Content-Based Filtering":

        st.subheader("Content-Based Recommendations")
        with st.expander("Apa itu Content-Based Filtering?"):
            st.write("""
            Content-Based Filtering menyarankan tempat yang mirip dengan tempat yang dipilih berdasarkan fitur-fitur seperti nama, kategori. Teknik ini menggunakan CountVectorizer untuk mengubah data teks menjadi vektor numerik dan menghitung kesamaan menggunakan Cosine Similarity. Cocok untuk pengguna baru, tetapi bisa kurang beragam.
            """)
        selected_place = st.text_input("Masukan Nama atau Kategori Tempat:")

        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi"):
            recommendations = content_based_recommendation(

                name=selected_place,

                cosine_sim=cosine_sim,

                items=merged_data,

                n=num_recommendations

            )

            st.write("Rekomendasi Tempat:")

            st.dataframe(recommendations)

    if selected_model == "Content-Based Filtering+":

        st.subheader("Content-Based Recommendations+")
        with st.expander("Apa itu Content-Based Filtering+?"):
            st.write("""
            Content-Based Filtering+ menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk memberikan bobot pada istilah berdasarkan kepentingannya dalam dataset. Metode ini menangkap keunikan tempat lebih baik dibandingkan dengan CountVectorizer, menghasilkan rekomendasi yang lebih relevan dan akurat.
            """)

        selected_place = st.text_input("Masukan Konten Wisata:")

        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi"):
            recommendations = content_based_recommendation_tfidf(
                name=selected_place,
                cosine_sim_tfidf=cosine_sim_tfidf,  # Correct parameter name
                items=merged_data2,
                n=num_recommendations
            )

            st.write("Rekomendasi Tempat:")

            st.dataframe(recommendations)

    if selected_model == "Collaborative Filtering SVD":
        st.subheader("Collaborative Filtering with SVD")
        with st.expander("Apa itu Collaborative Filtering with SVD?"):
            st.write("""
                Collaborative Filtering dengan SVD (Singular Value Decomposition) adalah teknik matrix factorization yang mendekomposisi matriks interaksi pengguna-tempat menjadi faktor laten. Pendekatan ini efektif untuk dataset yang jarang, di mana banyak pengguna hanya memberikan rating untuk beberapa tempat.
                """)
        # Show user table
        display_user_table()
        # User input for User ID
        user_id = st.number_input("Masukan ID Pengguna:", min_value=1, step=1)
        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi"):
            if user_id not in tourism_rating['User_Id'].unique():
                st.error(f"User ID {user_id} is not found in the dataset!")
            else:
                recommendations = collaborative_filtering_svd(tourism_rating, user_id, n=num_recommendations)
                st.write("Berikut ini adalah Rekomendasi berdasarkan preferensi pengguna:")
                st.dataframe(recommendations)
# Item-Based Collaborative Filtering
    if selected_model == "Item-Based Collaborative Filtering":
        st.subheader("Item-Based Collaborative Filtering")
        st.write("Available Place IDs in the dataset:")
        st.write(tourism_rating['Place_Id'].unique())
        # Check unique Place IDs in both datasets
        rating_ids = tourism_rating['Place_Id'].unique()
        tourism_ids = tourism_with_id['Place_Id'].unique()

        # Find mismatched IDs
        missing_in_rating = set(tourism_ids) - set(rating_ids)
        missing_in_tourism = set(rating_ids) - set(tourism_ids)

        st.write("Place IDs missing in ratings data:", missing_in_rating)
        st.write("Place IDs missing in tourism data:", missing_in_tourism)
        with st.expander("Apa itu Item-Based Collaborative Filtering?"):
            st.write("""
            Item-Based Collaborative Filtering memberikan rekomendasi berdasarkan kesamaan antara item.
            """)
            # Show tourism table
        display_wisata_table()
        place_id_input = st.text_input("Masukkan ID Tempat:")
        num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

        if st.button("Rekomendasi "):
            if place_id_input:
                recommendations = item_based_recommendation(tourism_rating, place_id_input, tourism_with_id, n=num_recommendations)
                st.write("Rekomendasi Tempat Berdasarkan Item:")
                st.dataframe(recommendations)



# elif page == "Statistik":
with tab2:
    display_statistics()

# elif page == "Top 10 tempat terbaik":
with tab3:
    st.subheader("Top 10 tempat terbaik untuk dikunjungi di lamongan")
    with st.expander("Penjelasan Top 10 Tempat Terbaik"):
        st.write("""
        Halaman ini menampilkan 10 tempat terbaik untuk dikunjungi berdasarkan kombinasi rating rata-rata dan jumlah ulasan.
        Tempat dengan banyak ulasan biasanya lebih dapat dipercaya, sementara rating memberikan indikasi kualitas. 
        Sistem menggunakan metode weighted scoring untuk menyeimbangkan kedua faktor ini:

        - **Weighted Score**: Menggabungkan rating rata-rata dengan jumlah ulasan untuk menghitung skor tempat.
        - Tempat dengan skor tertinggi akan ditampilkan dalam tabel.

        Halaman ini dirancang untuk memberikan rekomendasi umum bagi pengguna yang ingin menjelajahi tempat-tempat paling populer.
        """)
    # Compute the top 10 places based on ratings and the number of reviews
    top_places = compute_top_places(tourism_with_id)
    st.write("Top 10 tempat terbaik (Dibobotkan):")
    st.dataframe(top_places)

add_footer()