import pickle
import streamlit as st

st.title("Spam Detector using Naïve Bayes")
st.write("Enter a message below to check if it's spam or not.")

# Load both the trained vectorizer and model
with open(r"G:\My Projects\Predictive\spam-detector\spam_classifier_model.pkl", "rb") as f:
    vectorizer, loaded_model = pickle.load(f)  # Unpack the tuple correctly

# User input
message = st.text_area("Enter your message:")

if st.button("Check"):
    if message.strip():
        # Transform the message using the loaded vectorizer
        message_transformed = vectorizer.transform([message])
        
        # Make prediction
        prediction = loaded_model.predict(message_transformed)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("Please enter a message to check.")