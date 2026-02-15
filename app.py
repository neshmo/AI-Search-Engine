import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
from dotenv import load_dotenv
import os


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data(): 
    file_path = "female_products_with_embeddings.csv"
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['title', 'brand', 'category', 'product_description'])
    if 'embedding' in df.columns:
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    else:
        df['embedding'] = df.apply(lambda row: get_embedding(row), axis=1)
        df.to_csv(file_path, index=False)
    return df

def get_embedding(row):
    text = f"{row['title']} {row['brand']} {row['category']} {row['product_description']}"
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_query(query):
    response = openai.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_fashion(query, top_n=20):
    disallowed_keywords = {
        "jacket", "jackets",
        "hoodie", "hoodies",
        "skirt", "skirts",
        "shorts",
        "sweatshirt", "sweatshirts"
    }

    query_words = set(query.lower().split())
    if disallowed_keywords & query_words:
        fallback_results = data.sample(top_n)
        return "blocked", fallback_results

    query_embedding = embed_query(query)
    data['similarity'] = data['embedding'].apply(lambda emb: cosine_similarity([query_embedding], [emb])[0][0])
    top_results = data.sort_values(by='similarity', ascending=False).head(100)  # larger sample for filtering

    filtered_results = top_results[top_results.apply(
        lambda row: any(
            kw in (str(row['title']) + str(row['category']) + str(row['product_description'])).lower()
            for kw in query_words
        ), axis=1
    )]

    if not filtered_results.empty:
        return "matched", filtered_results.head(top_n)
    else:
        return "no_match", pd.DataFrame()

st.title("Women's Fashion Search Engine")

query = st.text_input("Search for dresses, tops, or t-shirts:")

data = load_data()

if query:
    status, results = search_fashion(query)

    if status == "blocked":
        st.warning("No Products Found, Shop from similar Categories.")
    elif status == "no_match":
        st.write("No results found.")

    if not results.empty:
        for _, row in results.iterrows():
            if pd.notna(row.get('image')) and row.get('image') != '':
                st.image(row['image'], caption=row['title'], use_column_width=True)
            st.write(f"*Brand:* {row['brand']}")
            st.write(f"*Category:* {row['category']}")
            st.write(f"*Colour:* {row['colour']}")
            st.write(f"*Price:* ₹{row['selling_price']}") 
            st.write(f"*Description:* {row['product_description']}")
            st.write(f"[View Product]({row['link']})")
            st.write("---")
