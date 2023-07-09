import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Book Recommender System')

@st.cache_data(experimental_allow_widgets=True)
def load_data(url):
    df = pd.read_csv(url)
    return df

df = load_data("C:\\Users\\radhakrishna.sp\\OneDrive - Impelsys\\Desktop\\demo_rec_eng_data.csv").drop('Unnamed: 0', axis=1)

# Function to calculate cosine similarity and cache the result
@st.cache(allow_output_mutation=True)
def calculate_cosine_similarity(_tfidf_matrix):
    cosine_sim = cosine_similarity(_tfidf_matrix)
    return pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])

# Function to recommend books based on similarity
def recommendations(book, similarity_df, items, k=10):
    ix = similarity_df.loc[:, book].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_df.columns[ix[-1:-(k + 2):-1]]
    closest = closest.drop(book, errors='ignore')
    recommend = pd.DataFrame(closest).merge(items).head(k).sort_values('title')
    return recommend

def main():
    # Calculate cosine similarity if not already cached
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=5, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['desc_gen'])
    cosine_sim_df = calculate_cosine_similarity(tfidf_matrix)

    # Set page title
    # st.title("Book Recommendation System")

    # Select book from dropdown or input text
    book_selection = st.selectbox("Select or search for a book:", df['title'], key='book_selection')
    if book_selection == 'Search for a book...':
        book = st.text_input("Enter the book name:", key='book_input')
    else:
        book = book_selection

    # Check if a book is selected
    if book:
        # Get recommendations for selected book
        recommended_books = recommendations(book, cosine_sim_df, df).sort_values('title')

        st.subheader("Similar books:")
        images_container = st.container()
        with images_container:
            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(recommended_books.iterrows()):
                response = requests.get(row['coverImg'])
                img = Image.open(BytesIO(response.content))

                if i % 3 == 0:
                    image_col = col1
                elif i % 3 == 1:
                    image_col = col2
                else:
                    image_col = col3

                with image_col:
                    st.image(img, caption=row['title'], width=200)

if __name__ == '__main__':
    main()
