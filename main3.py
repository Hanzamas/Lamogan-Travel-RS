import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# Load datasets
@st.cache
def load_data():
    tourism_rating = pd.read_csv("data/tourism_rating.csv", encoding="Windows-1252")
    tourism_with_id = pd.read_csv("data/tourism_with_id.csv", encoding="Windows-1252")
    user_data = pd.read_csv("data/user.csv", encoding="Windows-1252")
    return tourism_rating, tourism_with_id, user_data

tourism_rating, tourism_with_id, user_data = load_data()

# Clean and preprocess data
def preprocess_data():
    tourism_with_id['Jumlah Ulasan'] = tourism_with_id['Jumlah Ulasan'].str.replace(',', '').astype(int)
    merged_data = pd.merge(tourism_rating, tourism_with_id, on="Place_Id")
    merged_data['content'] = merged_data[
        ['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating', 'Fasilitas']
    ].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)
    return merged_data

merged_data = preprocess_data()

# Compute TF-IDF and Cosine Similarity
@st.cache
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['Place_Name']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(merged_data)

# Content-Based Recommendation
def content_based_recommendation(data, title, n=5):
    if title not in indices:
        st.error(f"The place '{title}' is not found in the dataset!")
        return pd.DataFrame()

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    sim_scores = cosine_sim[idx].flatten()
    sim_scores = [(i, score) for i, score in enumerate(sim_scores) if i != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n]

    place_indices = [i[0] for i in sim_scores]
    return data.iloc[place_indices][['Place_Name', 'Category', 'City', 'Rating', 'Description']]

# Collaborative Filtering
def collaborative_filtering(data, user_id, n=5):
    reader = Reader(rating_scale=(0.5, 5))
    rating_data = data[['User_Id', 'Place_Id', 'Place_Ratings']]
    dataset = Dataset.load_from_df(rating_data, reader)

    svd = SVD()
    trainset = dataset.build_full_trainset()
    svd.fit(trainset)

    if user_id not in rating_data['User_Id'].unique():
        st.error(f"User ID {user_id} is not found in the dataset!")
        return pd.DataFrame()

    all_places = data['Place_Id'].unique()
    predictions = [(place, svd.predict(user_id, place).est) for place in all_places]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    recommended_places = [place[0] for place in predictions]
    return data[data['Place_Id'].isin(recommended_places)][['Place_Name', 'Category', 'City', 'Rating', 'Description']]

# Streamlit UI
st.title("Travel Recommendation System")
st.sidebar.header("Recommendation Options")
selected_model = st.sidebar.selectbox(
    "Select Recommendation Model:",
    ["Content-Based Filtering", "Collaborative Filtering"]
)

if selected_model == "Content-Based Filtering":
    st.subheader("Content-Based Recommendations")
    selected_place = st.selectbox("Select a Place:", merged_data['Place_Name'].unique())
    num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Recommend Based on Content"):
        recommendations = content_based_recommendation(merged_data, selected_place, num_recommendations)
        st.write("Here are the top recommended places:")
        st.dataframe(recommendations)

elif selected_model == "Collaborative Filtering":
    st.subheader("Collaborative Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Recommend Based on User Ratings"):
        recommendations = collaborative_filtering(tourism_rating, user_id, num_recommendations)
        st.write("Here are the top recommendations based on your preferences:")
        st.dataframe(recommendations)
