
import streamlit as st
import pickle

st.title("Spam Detector using Naïve Bayes")
st.write("Enter a message below to check if it's spam or not.")

# Load the trained model (Make sure to provide the correct path to the model)
loaded_model = pickle.load(open(r"G:\My Projects\Predictive\spam-detector\spam_classifier_model.pkl", "rb"))

# User input
message = st.text_area("Enter your message:")

if st.button("Check"):
    if message.strip():
        # Make prediction
        prediction = loaded_model.predict([message])[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("Please enter a message to check.")
