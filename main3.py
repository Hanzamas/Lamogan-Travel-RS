import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Buat daftar stop words bahasa Indonesia menggunakan Sastrawi
factory = StopWordRemoverFactory()
indonesian_stop_words = factory.get_stop_words()

# Load datasets
@st.cache_data
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

    # Create 'content' column using relevant fields
    merged_data['content'] = merged_data[
        ['Place_Name', 'Category', 'Description']
    ].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)

    return merged_data

merged_data = preprocess_data()

# Compute TF-IDF and Cosine Similarity
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words=indonesian_stop_words)
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

    # Ambil indeks tempat yang dipilih
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    # Hitung skor kesamaan untuk semua tempat
    sim_scores = cosine_sim[idx].flatten()

    # Buat DataFrame skor kemiripan
    similarity_df = pd.DataFrame({
        'index': range(len(sim_scores)),
        'similarity': sim_scores
    })

    # Tambahkan informasi tempat ke DataFrame skor kemiripan
    similarity_df = similarity_df.merge(data[['Place_Name', 'Category']], left_on='index', right_index=True)

    # Filter tempat kecuali tempat input
    filtered_df = similarity_df[similarity_df['index'] != idx]

    # Urutkan berdasarkan skor kemiripan dan ambil n rekomendasi teratas
    top_indices = filtered_df.sort_values('similarity', ascending=False).head(n)['index']
    recommendations = data.iloc[top_indices][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]

    return recommendations

# Content-Based Filtering+
def content_based_recommendation_plus(data, title, n=5, min_reviews=0, min_rating=0):
    if title not in indices:
        st.error(f"The place '{title}' is not found in the dataset!")
        return pd.DataFrame()

    # Ambil indeks tempat yang dipilih
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    idx = int(idx)

    # Hitung skor kesamaan untuk semua tempat
    sim_scores = cosine_sim[idx].flatten()

    # Buat DataFrame skor kemiripan
    similarity_df = pd.DataFrame({
        'index': range(len(sim_scores)),
        'similarity': sim_scores
    })

    # Tambahkan informasi tempat ke DataFrame skor kemiripan
    similarity_df = similarity_df.merge(data[['Place_Name', 'Category', 'Rating', 'Jumlah Ulasan']], left_on='index', right_index=True)

    # Filter tempat kecuali tempat input dan tambahkan kriteria popularitas serta rating minimum
    filtered_df = similarity_df[
        (similarity_df['index'] != idx) &
        (similarity_df['Jumlah Ulasan'] >= min_reviews) &
        (similarity_df['Rating'] >= min_rating)
    ]

    # Urutkan berdasarkan skor kemiripan dan ambil n rekomendasi teratas
    top_indices = filtered_df.sort_values('similarity', ascending=False).head(n)['index']
    recommendations = data.iloc[top_indices][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]

    return recommendations


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

    visited_places = data[data['User_Id'] == user_id]['Place_Id'].tolist()
    all_places = data['Place_Id'].unique()
    unvisited_places = [place for place in all_places if place not in visited_places]

    if not unvisited_places:
        fallback_recommendations = tourism_with_id.sort_values(by=['Rating', 'Jumlah Ulasan'], ascending=[False, False])
        return fallback_recommendations[['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']].head(n)

    predictions = [(place, svd.predict(user_id, place).est) for place in unvisited_places]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    recommended_places = [place[0] for place in predictions]
    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(recommended_places)][['Place_Name', 'Category', 'City', 'Rating', 'Price', 'Description']]

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
page = st.sidebar.radio("Go to:", ["Recommendation System", "Statistics"])

if page == "Recommendation System":
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

        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend Based on Content"):
            recommendations = content_based_recommendation(

                merged_data,

                title=selected_place,

                n=num_recommendations

            )

            st.write("Here are the top recommended places:")

            st.dataframe(recommendations)


    elif selected_model == "Collaborative Filtering":

        st.subheader("Collaborative Recommendations")

        user_id = st.number_input("Enter User ID:", min_value=1, step=1)

        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend Based on User Ratings"):
            recommendations = collaborative_filtering(

                tourism_rating,

                user_id,

                n=num_recommendations

            )

            st.write("Here are the top recommendations based on your preferences:")

            st.dataframe(recommendations)

    elif selected_model == "Content-Based Filtering+":
        st.subheader("Content-Based Recommendations+")

        selected_place = st.selectbox("Select a Place (Required):", merged_data['Place_Name'].unique())
        min_reviews = st.number_input("Minimum Number of Reviews (Optional):", min_value=0, step=1, value=0)
        min_rating = st.slider("Minimum Rating (Optional):", min_value=0.0, max_value=5.0, step=0.1, value=0.0)
        num_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

        if st.button("Recommend Based on Content+"):
            recommendations = content_based_recommendation_plus(
                merged_data,
                title=selected_place,
                n=num_recommendations,
                min_reviews=min_reviews,
                min_rating=min_rating
            )
            st.write("Here are the top recommended places based on enhanced criteria:")
            st.dataframe(recommendations)

elif page == "Statistics":
    display_statistics()
