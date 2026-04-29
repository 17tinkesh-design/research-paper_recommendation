import os
import urllib.request
import streamlit as st
import torch
from sentence_transformers import util, SentenceTransformer
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# --- 1. AUTOMATIC MODEL DOWNLOADER ---
@st.cache_resource
def download_models():
    """Downloads large model files from GitHub Releases if they don't exist."""
    os.makedirs('models', exist_ok=True)
    
    # Using your exact GitHub username and repository
    base_url = "https://github.com/17tinkesh-design/research_paper_recommendation/releases/download/v1.0/"
    
    # List of massive files that had to be uploaded to Releases
    files_to_download = {
        'models/embeddings.pkl': base_url + 'embeddings.pkl',
        'models/model.h5': base_url + 'model.h5',
        'models/sentences.pkl': base_url + 'sentences.pkl' # Included just in case this was also large
    }
    
    for file_path, url in files_to_download.items():
        if not os.path.exists(file_path):
            with st.spinner(f"Downloading {file_path} (this happens only once)..."):
                try:
                    urllib.request.urlretrieve(url, file_path)
                except Exception as e:
                    st.error(f"Could not download {file_path}. Make sure it is uploaded to your v1.0 GitHub Release!")

# Run the download check before trying to load anything
download_models()


# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Research AI", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-text {
        color: #888888;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- 3. LOAD MODELS (CACHED) ---
@st.cache_resource
def load_all_components():
    # A. Load Recommendation Engine
    rec_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
        sentences = pickle.load(open('models/sentences.pkl', 'rb'))
    except Exception:
        embeddings, sentences = None, None
    
    # B. Attempt to Load Prediction Model
    try:
        loaded_model = keras.models.load_model("models/model.h5")
        with open("models/text_vectorizer_config.pkl", "rb") as f:
            config = pickle.load(f)
        
        config.pop('batch_input_shape', None)
        config.pop('dtype', None) 
        
        vectorizer = TextVectorization.from_config(config)
        
        with open("models/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
            
        with open("models/text_vectorizer_weights.pkl", "rb") as f:
            weights = pickle.load(f)

        vectorizer.set_vocabulary(vocab) 
        vectorizer.set_weights(weights)
        return rec_model, embeddings, sentences, loaded_model, vectorizer, vocab
    except Exception:
        # Fallback if smaller files are missing
        return rec_model, embeddings, sentences, None, None, None

rec_model, embeddings, sentences, loaded_model, loaded_text_vectorizer, loaded_vocab = load_all_components()


# --- 4. LOGIC FUNCTIONS ---
def get_recommendations(title_input):
    if embeddings is None or sentences is None:
        return ["Error: Embeddings file not found. Check GitHub Releases."]
    query_emb = rec_model.encode(title_input)
    scores = util.cos_sim(embeddings, query_emb)
    top_results = torch.topk(scores, dim=0, k=5, sorted=True)
    return [sentences[idx.item()] for idx in top_results.indices]

def predict_subject_fallback(abstract):
    keywords = {"neural": "Computer Science", "network": "Deep Learning", "algorithm": "Mathematics"}
    found = [val for key, val in keywords.items() if key in abstract.lower()]
    return found if found else ["General Research"]


# --- 5. STREAMLIT UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3364/3364350.png", width=60)
    st.header("About the System")
    st.write("This engine leverages Natural Language Processing to find semantic similarities between research papers and predict core subject areas based on abstract data.")
    st.divider()
    st.caption("System Status: Online 🟢")
    if loaded_model:
        st.caption("Deep Learning Model: Active")
    else:
        st.caption("Deep Learning Model: Fallback Mode")

st.markdown('<p class="main-header">🎓 Research Paper Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter your paper details below to discover related literature and analyze subjects.</p>', unsafe_allow_html=True)

with st.form("analysis_form"):
    st.subheader("Data Input")
    title_search = st.text_input("Enter Paper Title:", placeholder="e.g., Attention Is All You Need")
    abstract_search = st.text_area("Paste Abstract Content:", height=150, placeholder="Paste the abstract text here...")
    
    submit_button = st.form_submit_button("🚀 Analyze Research Data", use_container_width=True)

if submit_button:
    if not title_search and not abstract_search:
        st.warning("⚠️ Please provide either a title or an abstract to analyze.")
    else:
        with st.spinner("Processing text and generating insights..."):
            res_col1, res_col2 = st.columns(2)
            
            if title_search:
                with res_col1:
                    st.success("### 📚 Top Recommendations")
                    recommendations = get_recommendations(title_search)
                    for i, r in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {r}")
            
            if abstract_search:
                with res_col2:
                    st.info("### 🎯 Predicted Subject Areas")
                    if loaded_model:
                        vec = loaded_text_vectorizer([abstract_search])
                        preds = loaded_model.predict(vec)
                        st.write("Subject identified by Deep Learning model.")
                    else:
                        predictions = predict_subject_fallback(abstract_search)
                        for pred in predictions:
                            st.markdown(f"- **{pred}**")
