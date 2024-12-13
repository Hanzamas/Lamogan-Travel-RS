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
    user_data = pd.read_csv("user.csv", encoding="Windows-1252")
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
def content_based_recommendation(data, title, min_price=None, max_price=None, min_rating=None, n=5):
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
    recommendations = data.iloc[place_indices][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]

    if min_price:
        recommendations = recommendations[recommendations['Price'] >= min_price]
    if max_price:
        recommendations = recommendations[recommendations['Price'] <= max_price]
    if min_rating:
        recommendations = recommendations[recommendations['Rating'] >= min_rating]

    return recommendations

# Collaborative Filtering
def collaborative_filtering(data, user_id, min_price=None, max_price=None, min_rating=None, n=5):
    reader = Reader(rating_scale=(0.5, 5))
    rating_data = data[['User_Id', 'Place_Id', 'Place_Ratings']]
    dataset = Dataset.load_from_df(rating_data, reader)

    svd = SVD()
    trainset = dataset.build_full_trainset()
    svd.fit(trainset)

    if user_id not in rating_data['User_Id'].unique():
        st.error(f"User ID {user_id} is not found in the dataset!")
        return pd.DataFrame()

    visited_places = data[data['User_Id'] == user_id]['Place_Id'].tolist()
    all_places = data['Place_Id'].unique()
    unvisited_places = [place for place in all_places if place not in visited_places]

    predictions = [(place, svd.predict(user_id, place).est) for place in unvisited_places]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    recommended_places = [place[0] for place in predictions]
    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_places)][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]

    if min_price:
        recommendations = recommendations[recommendations['Price'] >= min_price]
    if max_price:
        recommendations = recommendations[recommendations['Price'] <= max_price]
    if min_rating:
        recommendations = recommendations[recommendations['Rating'] >= min_rating]

    return recommendations

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

# Streamlit UI
st.title("Travel Recommendation System")
st.sidebar.header("Recommendation Options")
selected_model = st.sidebar.selectbox(
    "Select Recommendation Model:",
    ["Simple Recommendation", "Content-Based Filtering", "Collaborative Filtering"]
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

elif selected_model == "Content-Based Filtering":
    st.subheader("Content-Based Recommendations")
    selected_place = st.selectbox("Select a Place (Required):", merged_data['Place_Name'].unique())
    min_price = st.number_input("Minimum Price (Optional):", min_value=0, step=1, value=0)
    max_price = st.number_input("Maximum Price (Optional):", min_value=0, step=1, value=0)
    min_rating = st.slider("Minimum Rating (Optional):", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
    num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Recommend Based on Content"):
        recommendations = content_based_recommendation(
            merged_data,
            title=selected_place,
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price > 0 else None,
            min_rating=min_rating if min_rating > 0 else None,
            n=num_recommendations
        )
        st.write("Here are the top recommended places:")
        st.dataframe(recommendations)

elif selected_model == "Collaborative Filtering":
    st.subheader("Collaborative Recommendations")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    min_price = st.number_input("Minimum Price (Optional):", min_value=0, step=1, value=0)
    max_price = st.number_input("Maximum Price (Optional):", min_value=0, step=1, value=0)
    min_rating = st.slider("Minimum Rating (Optional):", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
    num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Recommend Based on User Ratings"):
        recommendations = collaborative_filtering(
            tourism_rating,
            user_id,
            min_price=min_price if min_price > 0 else None,
            max_price=max_price if max_price > 0 else None,
            min_rating=min_rating if min_rating > 0 else None,
            n=num_recommendations
        )
        st.write("Here are the top recommendations based on your preferences:")
        st.dataframe(recommendations)
