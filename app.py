# Import libraries - regular Python imports first
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="✉️",
    layout="wide"
)

# Download NLTK resources (with caching)
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        st.stop()

download_nltk_resources()

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    st.warning("Required NLTK data not found. Downloading...")
    download_nltk_resources()

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocesses text by:
    1. Converting to lowercase
    2. Removing special characters
    3. Removing numbers
    4. Removing URLs
    5. Removing email addresses
    6. Tokenizing
    7. Removing stopwords
    8. Lemmatizing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer and get stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back to string
    return ' '.join(tokens)

# Load model and vectorizer (with caching)
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_spam_classifier.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'best_spam_classifier.joblib' and 'tfidf_vectorizer.joblib' are in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Prediction function
def predict_email(text):
    cleaned_text = preprocess_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)
    proba = model.predict_proba(text_vec)
    
    return {
        'prediction': 'spam' if prediction[0] == 1 else 'ham',
        'spam_probability': proba[0][1],
        'ham_probability': proba[0][0]
    }

# UI Components
st.title("✉️ Spam Email Classifier")
st.markdown("""
This app predicts whether an email is spam or ham (not spam) using a trained machine learning model.
""")

# Input options
input_option = st.radio(
    "Choose input method:",
    ("Enter text", "Load sample email")
)

email_text = ""

if input_option == "Enter text":
    email_text = st.text_area("Paste the email content here:", height=200)
else:
    sample_spam = """
    Subject: Win a free iPhone!
    Congratulations! You've been selected to win a brand new iPhone. 
    Click here to claim your prize now: http://fakewebsite.com/win
    Don't miss this limited time offer!
    """
    
    sample_ham = """
    Subject: Meeting tomorrow
    Hi team,
    
    Just a reminder about our quarterly planning meeting tomorrow at 10am in Conference Room B.
    Please bring your project updates and any questions you have.
    
    Best regards,
    John
    """
    
    sample_choice = st.radio("Choose sample type:", ("Spam", "Ham"))
    email_text = st.text_area("Sample email:", 
                             value=sample_spam if sample_choice == "Spam" else sample_ham, 
                             height=200)

# Prediction button
if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter some text or load the sample email.")
    else:
        with st.spinner("Analyzing the email..."):
            try:
                result = predict_email(email_text)
                
                st.subheader("Prediction Results")
                
                # Display prediction with color
                if result['prediction'] == 'spam':
                    st.error(f"Prediction: **{result['prediction']}** 🚨")
                else:
                    st.success(f"Prediction: **{result['prediction']}** ✅")
                
                # Display probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Spam Probability", f"{result['spam_probability']:.2%}")
                with col2:
                    st.metric("Ham Probability", f"{result['ham_probability']:.2%}")
                
                # Show a nice progress bar
                st.progress(result['spam_probability'])
                st.caption("Spam confidence level")
                
                # Show cleaned text (for debugging/transparency)
                with st.expander("Show cleaned text"):
                    st.text(preprocess_text(email_text))
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

# Add some info in the sidebar
with st.sidebar:
    st.markdown("""
    **About this app:**
    - Uses TF-IDF for text vectorization
    - Includes comprehensive text preprocessing:
      - Lowercasing
      - URL/email removal
      - Special character removal
      - Stopword removal
      - Lemmatization
    - Model: Best performing model from your training
    """)
    
    st.markdown("""
    **How to use:**
    1. Paste email text or load sample
    2. Click Predict button
    3. View results
    """)

# Optional: Add footer
st.markdown("---")
st.caption("Spam Classifier App | Made with Streamlit")