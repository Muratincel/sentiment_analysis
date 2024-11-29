import joblib

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to predict sentiment
def predict_sentiment(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"

# in order not to run externally, only when called as predict.py manually
if __name__ == '__main__':
    while True:
        review = input("Enter a review (or 'exit' to quit): ")
        if review.lower() == 'exit':
            break
        print(predict_sentiment(review))  # Or your function for prediction
