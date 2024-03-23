import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stemming = PorterStemmer()


with open('model.pkl', 'rb') as model_file:
    loaded_lr = pickle.load(model_file)


with open('vector.pkl', 'rb') as file:
    loaded_cv = pickle.load(file)

# Function to preprocess input text
def preprocess_text(text):
    preprocessed_text = ""
    sentences = nltk.sent_tokenize(text)
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        words = nltk.word_tokenize(sentences[i])
        words = [lemmatizer.lemmatize(word,pos='v') for word in words if word not in set(stopwords.words('english'))]
        words  = [stemming.stem(word) for word in words]
        preprocessed_text += " ".join(words) + " "  # Add a space to separate sentences
    return preprocessed_text.strip()


# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = loaded_cv.transform([preprocessed_text])
    prediction = loaded_lr.predict(vectorized_text)
    return prediction[0]

# Streamlit app
def main():
    st.title('Sentiment Analysis App')
    st.write("Please note: Input text is preferable in English for better accuracy. Support for other languages will be added soon.")
    # Text input for user to enter text for sentiment analysis
    user_input = st.text_input('Enter text for sentiment analysis:', '')

    if st.button('Analyze Sentiment'):
        if user_input.strip() == '':
            st.error('Please enter some text.')
        else:
            sentiment_label = {1: 'Positive', 2: 'Negative', 3: 'Neutral', 4: 'Irrelevant'}
            prediction = predict_sentiment(user_input)
            st.success(f'The sentiment of the text is: {sentiment_label[prediction]}')

if __name__ == '__main__':
    main()
