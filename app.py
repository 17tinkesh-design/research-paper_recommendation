import streamlit as st
import torch
from sentence_transformers import util, SentenceTransformer
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Research AI", page_icon="📚", layout="wide")

# Optional: A tiny bit of custom CSS to make the title pop
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

# 2. LOAD MODELS (CACHED)
@st.cache_resource
def load_all_components():
    # A. Load Recommendation Engine
    rec_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
    sentences = pickle.load(open('models/sentences.pkl', 'rb'))
    
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

        # C. Patch for the Vocab Issue
        vectorizer.set_vocabulary(vocab) 
        vectorizer.set_weights(weights)
        return rec_model, embeddings, sentences, loaded_model, vectorizer, vocab
    except Exception:
        # Fallback: App will still load, but Subject Prediction will use a simpler method
        return rec_model, embeddings, sentences, None, None, None

# Run the loader
rec_model, embeddings, sentences, loaded_model, loaded_text_vectorizer, loaded_vocab = load_all_components()

# 3. LOGIC FUNCTIONS
def get_recommendations(title_input):
    query_emb = rec_model.encode(title_input)
    scores = util.cos_sim(embeddings, query_emb)
    top_results = torch.topk(scores, dim=0, k=5, sorted=True)
    return [sentences[idx.item()] for idx in top_results.indices]

def predict_subject_fallback(abstract):
    keywords = {"neural": "Computer Science", "network": "Deep Learning", "algorithm": "Mathematics"}
    found = [val for key, val in keywords.items() if key in abstract.lower()]
    return found if found else ["General Research"]

# 4. STREAMLIT UI

# Sidebar for professional framing
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3364/3364350.png", width=60) # Placeholder generic graduation icon
    st.header("About the System")
    st.write("This engine leverages Natural Language Processing to find semantic similarities between research papers and predict core subject areas based on abstract data.")
    st.divider()
    st.caption("System Status: Online 🟢")
    if loaded_model:
        st.caption("Deep Learning Model: Active")
    else:
        st.caption("Deep Learning Model: Fallback Mode")

# Main Header
st.markdown('<p class="main-header">🎓 Research Paper Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter your paper details below to discover related literature and analyze subjects.</p>', unsafe_allow_html=True)

# Using a form bundles the inputs together and looks much cleaner
with st.form("analysis_form"):
    st.subheader("Data Input")
    title_search = st.text_input("Enter Paper Title:", placeholder="e.g., Attention Is All You Need")
    abstract_search = st.text_area("Paste Abstract Content:", height=150, placeholder="Paste the abstract text here...")
    
    submit_button = st.form_submit_button("🚀 Analyze Research Data", use_container_width=True)

# Results Section
if submit_button:
    if not title_search and not abstract_search:
        st.warning("⚠️ Please provide either a title or an abstract to analyze.")
    else:
        # Spinner gives a professional feel while processing
        with st.spinner("Processing text and generating insights..."):
            
            # Create two columns for the outputs
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