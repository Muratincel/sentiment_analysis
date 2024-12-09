## Our only essential files&folder;

app.py: The Flask application.

predict.py: Prediction logic.

logistic_regression_model.joblib and tfidf_vectorizer.joblib.

templates/ folder: Contains your index.html.

## How this works

python -m venv venv

venv\scripts\activate

pip install -r requirements.txt

run app.py

## The files for training data;

load_data.py: just to loading the data as texts and labels, creating preprocessed_data.csv by using preprocess.py

preprocess.py: shaping the data as a processable format, lower, special cases etc. by using nltk module

vectorize.py: creating the dataframe (df) by loading the preprocessed_data.csv

load_nltk.py: jus to download nltk dependencies

train_model.py: creating logistic_regression_model.joblib and tfidf_vectorizer.joblib files by training data. basicly creating the model


## not clear ? by ai then :)

load_data.py: Loads the data as texts and labels and creates preprocessed_data.csv by calling preprocess.py.

preprocess.py: Processes the data into a format suitable for modeling, including text cleaning, lowercasing, and handling special cases using the nltk module.

vectorize.py: Loads preprocessed_data.csv and creates a DataFrame (df) for further analysis and vectorization.

load_nltk.py: Downloads necessary nltk dependencies for text processing.

train_model.py: Trains the logistic regression model and saves the trained model as logistic_regression_model.joblib and the TF-IDF vectorizer as tfidf_vectorizer.joblib.

*training data is taken from: https://ai.stanford.edu/~amaas/data/sentiment/
