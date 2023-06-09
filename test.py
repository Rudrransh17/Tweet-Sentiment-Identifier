from nltk.tokenize import word_tokenize # Tokenizer to make tokens
from nltk.stem import PorterStemmer # Stemmer to perform stemming on tokens
from nltk.corpus import stopwords # Stopwords corpus from nltk library
import pickle
import numpy as np
import re
import tensorflow as tf
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set of stopwords
stopwords = set(stopwords.words('english'))

# Stemmer object
stemmer = PorterStemmer()

# Function to preprocess input tweets and return tokens
def preprocess_text(text):
    # Remove '@user' tags
    text = re.sub(r'@[^\s]+', '', text)

    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)

    # Lowercasing
    text = text.lower()

    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)

    # Stopword removal and stemming
    tokens = word_tokenize(text)
    # tokens = [token for token in tokens if token not in stopwords]
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the saved model
model = tf.keras.models.load_model('trained_model.h5')

# Get the input sentence
input_text = input("Enter a sentence: ")

# Preprocess the input sentence
input_tokens = preprocess_text(input_text)

# Transform the preprocessed tokens into a vector
input_vector = vectorizer.transform([input_tokens]).toarray()

# Make the prediction using the loaded model
prediction = np.argmax(model.predict([input_vector]), axis=1)

if prediction[0] == 4:
    sentiment = 'Positive'
elif prediction[0] == 2:
    sentiment = 'Neutral'
elif prediction[0] == 0:
    sentiment = 'Negative'
# Print the predicted output
print("Sentiment:", sentiment)
