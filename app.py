import pickle
import streamlit as st 
import numpy as np 

st.header("Book Recommendation System with KNN")
model = pickle.load(open('model.pkl', 'rb'))
bookname = pickle.load(open('bookname.pkl', 'rb'))
new_df = pickle.load(open('new_df.pkl', 'rb'))
df_pivot = pickle.load(open('df_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_name = []
    id_idx = []
    poster_url = []
    
    for book_id in suggestion:
        book_name.append(df_pivot.index[book_id])

    for name in book_name[0]:
        id = np.where(new_df['book_title'] == name)[0][0]
        id_idx.append(id)

    for idx in id_idx:
        url = new_df.iloc[idx]['image_url_l']
        poster_url.append(url)

    return poster_url

def recommend_books(bookname):
    book_list = []
    book_id = np.where(df_pivot.index == bookname)[0][0]
    distance, suggestion = model.kneighbors(df_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors = 6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = df_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list, poster_url

selected_books = st.selectbox(
    "Type or Select a Book", 
    bookname
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_books(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])

    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])

    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])

    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])


