import streamlit as st
import ast
import os
from PIL import Image
from io import BytesIO
import requests
from reserve import (get_download_url, parse_response, load_api_key, retrieve_book)
#from ipynb.fs.full.reserve import 
first_downloaded_save_path = os.getcwd()

openai_api_key, rapidai_api_key = load_api_key()

def show_if_present(value, render_func):
    if value not in (None, "", []):
        render_func(value)

def download_image(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    Image.open(BytesIO(response.content))

st.set_page_config(page_title="Audiobook Generator", layout="centered", initial_sidebar_state="collapsed")

# Inject custom CSS
st.markdown(
    """
    <style>
    /* 1. Main Background */
    [data-testid="stAppViewContainer"], .stApp, .main {
        background-color: #F5F5DC !important; 
    }
    
    /* 2. Header Background */
    [data-testid="stHeader"] {
        background-color: #F5F5DC !important;
    }
    
    /* 3. Text Styling (Title and labels) */
    h1, p, span, label {
        color: #3B3A36 !important; 
        font-family: 'Source Sans Pro', sans-serif;
    }

    /* 4. The Search Bar Container - Forced to light color */
    .stTextInput div[data-baseweb="input"], 
    .stTextInput div[data-baseweb="input"] .st-ae,
    .stTextInput div[data-baseweb="input"] .st-af {
        background-color: #FFFAF0 !important; 
        border: 1px solid #D3CFC1 !important;
        border-radius: 10px !important;
    }
    
    /* 5. The Input Text itself */
    .stTextInput input {
        color: #3B3A36 !important;
        background-color: transparent !important;
    }

    /* Remove the focus border glow for a cleaner look */
    .stTextInput div[data-baseweb="input"]:focus-within {
        border: 1px solid #3B3A36 !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title at the top
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; padding-bottom: 20px;'>Audiobook Generator</h1>", unsafe_allow_html=True)

# The Search Bar
search_query = st.text_input("Enter in any book. Include author and extension type (pdf, .epub) if you can.", placeholder="Search...")
if search_query:
    book_name, author, file_type = ast.literal_eval(parse_response(search_query))
    results = retrieve_book(book_name, author, file_type)
    print(f'results: {results[0]}')
    cols = st.columns(len(results))
    for col, book in zip(cols, results):
        # book_download_url = book.get('md5', '')
        # if book_download_url: 
        #     md5 = get_download_url(book_download_url)
        #     book_contents = download_book()
        with col: 
            title, author, img_url, date = book.get('title'), book.get('author'), book.get('imgUrl'), book.get('year')
            print(title, author, img_url, date)
            if title: 
                st.markdown(f'**{title}**')
            if img_url: 
                st.image(Image.open('/Users/sv/Desktop/audiobooks/IMG_4216.jpeg'), caption='SV')
                #st.image(download_image(img_url))
            if author:
                st.markdown(f"{author}")
            if date: 
                st.markdown(f"{date}")
            break
