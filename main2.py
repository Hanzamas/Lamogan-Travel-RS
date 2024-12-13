import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

# Load datasets
@st.cache
def load_data():
    tourism_rating = pd.read_csv("tourism_rating.csv", encoding="latin1")
    tourism_with_id = pd.read_csv("tourism_with_id.csv", encoding="latin1")
    user_data = pd.read_csv("user.csv", encoding="latin1")
    return tourism_rating, tourism_with_id, user_data

tourism_rating, tourism_with_id, user_data = load_data()

# Merge data for analysis
merged_data = pd.merge(tourism_rating, tourism_with_id, on="Place_Id")

# Prepare content for content-based filtering
@st.cache
def prepare_content_data(data):
    data['content'] = data[['Place_Name', 'Category', 'City', 'Description']].fillna('').apply(lambda x: ' '.join(x), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, pd.Series(data.index, index=data['Place_Name'])

cosine_sim, indices = prepare_content_data(merged_data)

# Sidebar filters
st.sidebar.header("Filter Options")
selected_page = st.sidebar.selectbox(
    "Select Page:",
    ["Recommendation System", "Data Statistics"]
)

selected_model = st.sidebar.selectbox(
    "Select Recommendation Model:",
    ["Simple Recommendation", "Content-Based Filtering", "Collaborative Filtering"],
    disabled=(selected_page != "Recommendation System")
)

selected_factors = st.sidebar.multiselect(
    "Select Factors for Simple Recommendation:",
    ["Rating", "Price", "Category", "City", "Reviews", "Transport"],
    default=["Rating"],
    disabled=(selected_page != "Recommendation System")
)

# Recommendation functions
def recommend_simple(data, factors):
    if "Rating" in factors:
        data = data.sort_values(by="Rating", ascending=False)
    if "Price" in factors:
        data = data.sort_values(by="Price", ascending=True)
    if "Category" in factors:
        categories = st.sidebar.multiselect(
            "Select Categories:", data["Category"].unique(), default=data["Category"].unique()
        )
        data = data[data["Category"].isin(categories)]
    if "City" in factors:
        cities = st.sidebar.multiselect(
            "Select Cities:", data["City"].unique(), default=data["City"].unique()
        )
        data = data[data["City"].isin(cities)]
    if "Reviews" in factors:
        data["Jumlah Ulasan"] = data["Jumlah Ulasan"].str.replace(",", "").astype(int)
        data = data.sort_values(by="Jumlah Ulasan", ascending=False)
    if "Transport" in factors:
        transport_options = st.sidebar.multiselect(
            "Select Transport Options:",
            ["mobil", "sepeda motor", "bus", "elf"],
            default=["mobil", "sepeda motor", "bus", "elf"]
        )
        data = data[data["Transportasi"].str.contains("|".join(transport_options), na=False)]
    return data.head(10)

def content_based_recommendation(data, title, n=5):
    if title not in indices:
        st.error("Selected place not found in the dataset!")
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    place_indices = [i[0] for i in sim_scores]
    return data.iloc[place_indices][["Place_Name", "Category", "City", "Rating"]]

def collaborative_filtering(data, user_id, n=5):
    reader = Reader(rating_scale=(0.5, 5))
    rating_data = data[['User_Id', 'Place_Id', 'Place_Ratings']]
    dataset = Dataset.load_from_df(rating_data, reader)
    svd = SVD()
    trainset = dataset.build_full_trainset()
    svd.fit(trainset)

    user_ratings = data[data['User_Id'] == user_id]
    if user_ratings.empty:
        st.error("No ratings found for the selected user!")
        return pd.DataFrame()

    rated_places = user_ratings['Place_Id'].tolist()
    all_places = data['Place_Id'].unique()
    unrated_places = [place for place in all_places if place not in rated_places]

    predictions = [
        (place, svd.predict(user_id, place).est) for place in unrated_places
    ]
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    recommended_places = [place[0] for place in top_predictions]
    return data[data['Place_Id'].isin(recommended_places)][["Place_Name", "Category", "City", "Rating"]]

# Data Statistics
if selected_page == "Data Statistics":
    st.title("Data Statistics")
    st.write("### Dataset Overview")
    st.dataframe(merged_data.head())

    st.write("### Summary Statistics")
    st.write(merged_data.describe(include='all'))

    st.write("### Distribution of Ratings")
    st.bar_chart(merged_data['Rating'].value_counts().sort_index())

    st.write("### Distribution of Categories")
    st.bar_chart(merged_data['Category'].value_counts())

    st.write("### Distribution of Prices")
    st.bar_chart(merged_data['Price'].value_counts().sort_index())

    st.write("### Top Cities by Number of Destinations")
    st.bar_chart(merged_data['City'].value_counts())

    st.write("### Map of Locations")
    if 'Latitude' in merged_data.columns and 'Longitude' in merged_data.columns:
        st.map(merged_data[['Latitude', 'Longitude']])
    else:
        st.warning("Location data (Latitude and Longitude) is not available in the dataset.")

    st.write("### User Data Statistics")
    st.write("#### User Overview")
    st.dataframe(user_data.head())

    st.write("#### Distribution of Age")
    st.bar_chart(user_data['Age'].value_counts().sort_index())

    st.write("#### Distribution of Gender")
    st.bar_chart(user_data['Gender'].value_counts())

    st.write("#### User Locations")
    st.bar_chart(user_data['Location'].value_counts().head(10))

# Display recommendations
if selected_page == "Recommendation System":
    st.title("Travel Recommendation System")
    st.write("This system provides recommendations based on different models.")

    if selected_model == "Simple Recommendation":
        st.write("### Simple Recommendations")
        recommended_places = recommend_simple(merged_data, selected_factors)
        st.dataframe(recommended_places[["Place_Name", "Rating", "Price", "Category", "City"]])

    elif selected_model == "Content-Based Filtering":
        st.write("### Content-Based Recommendations")
        selected_place = st.selectbox("Select a Place:", merged_data['Place_Name'].unique())
        if st.button("Recommend Based on Content"):
            recommendations = content_based_recommendation(merged_data, selected_place)
            st.dataframe(recommendations)

    elif selected_model == "Collaborative Filtering":
        st.write("### Collaborative Recommendations")
        user_id = st.number_input("Enter User ID:", min_value=1, step=1)
        if st.button("Recommend Based on User Ratings"):
            recommendations = collaborative_filtering(merged_data, user_id)
            st.dataframe(recommendations)
