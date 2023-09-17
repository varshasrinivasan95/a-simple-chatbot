## Simple Chatbot

This python code provides a simple implementation of an interactable chatbot that understands user message, and provides appropriate response. The chatbot is implemented using LogisticRegression model using sklearn package. 

## Code Flow

- The code starts by having a predefined set of intents that represents specfic category/labels associated with the user's message and the responses that can be returned from the Chatbot. 
- TfidfVectorizer is used for converting the user inputs into machine readable formats (numerical format).
- The LogisticRegression model is trained with the vectorized user inputs and the corresponding tags. 
- Streamlit python package is utilized for building web applications quickly
- Using the Streamlit functions, user input is received in the text box after which it is passed as input to the model for prediction 
- Whenever a user input is received, the model predicts the corresponding tag and returns respective response based on the tag using the predefined set of intents
- The response will be displayed in the text area using the Streamlit function
- If the user inputs any messages like 'goodbye' or 'bye', chatbot ends the conversation with a thank you message

## Code Comments
- Comments are given for every line of code for better understanding
- Comments explain the purpose of each package that are included, along with additional information about others

## Running the chatbot
- Use terminal for running the chatbot 
- Open the respective directory and provide the following command
- Command: streamlit run filename.py

## Reference

This project includes code sourced from the End to End Chatbot using Python (https://thecleverprogrammer.com/2023/03/27/end-to-end-chatbot-using-python/)
