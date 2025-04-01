from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle


st.title("Spam Detector using Naïve Bayes")
st.write("Enter a message below to check if it's spam or not.")

# Load the trained model (Make sure to provide the correct path to the model)
loaded_model = pickle.load(open("spam_classifier_model.pkl", "rb"))


# User input
message = st.text_area("Enter your message:")
vectorizer = TfidfVectorizer()
if st.button("Check"):
    if message.strip():
        # Make prediction
        
        message_transformed = vectorizer.transform([message])
        prediction = loaded_model.predict([message_transformed ])[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("Please enter a message to check.")
