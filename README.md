# Text Sentiment Identifier

This project aims to classify text sentiment using a machine learning model trained on the sentiment analysis dataset from Kaggle [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Implementation

### 1. Data Processing
- The data is read using the pandas library to load the dataset.
- A preprocess function is applied to the text data, which involves removing user tags, hyperlinks, lowercasing, punctuation, tokenization, and stemming. Stopwords are not removed to preserve important sentiment-related words like "not" and "no".
- The tokens and target values are stored in a separate file.

### 2. Vectorization
- A TF-IDF Vectorizer is created to convert the list of tokens into 50-dimensional vectors.
- The fitted vectorizer is saved as a pickle file (`vectorizer.pkl`) for future use.
- The vectors and their corresponding target values are saved in a separate file.

### 3. Model Training
- The data is split into training and validation sets.
- A sequential model is built using the Keras library.
- The model includes layers with batch normalization and dropout for regularization.
- The last layer uses softmax activation to output probabilities for each sentiment class.
- The model is trained for 10 epochs and saved as a h5 file (`trained_model.h5`) for future use.

### 4. Model Evaluation
- The trained model is evaluated on the validation dataset.
- The achieved metrics include accuracy, precision, recall, and F1-score, which approximately reach a value of 0.65.

## Files

- `app.py`: Flask application file that serves the sentiment prediction webpage.
- `index.html`: HTML file that displays the Text Sentiment Identifier webpage.
- `styles.css`: CSS file that contains the styling for the webpage.
- `trained_model.h5`: Saved model file in the .h5 format, which can be loaded to make predictions in the `app.py` file.
- `vectorizer.pkl`: Pickle file containing the fitted TF-IDF vectorizer, used to vectorize input text in the `app.py` file.
- `model.ipynb`: Jupyter Notebook file that contains the code used to train and evaluate the sentiment analysis model.

## Dependencies

The project requires the following dependencies to be installed:
- Python (version >= 3.6)
- pandas
- numpy
- scikit-learn
- tensorflow
- Keras
- Flask
- nltk

Please ensure that the necessary dependencies are installed.

## Dataset and Processed Data

Due to the large file sizes, the original sentiment analysis dataset and the processed data files are not included in this repository. Instead, the zipped versions of the files have been uploaded. Please unzip the files before running the project.

## How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the `app.py` file.
4. Access the Text Sentiment Identifier webpage on `localhost:5000` in your web browser.
