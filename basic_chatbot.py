# ================================================
# Section 1: Importing Libraries
# ================================================

# os package for communicating with our system to know things like where files/folders are, creating/ renaming files, etc.
import os

# Checks if a website is trustworthy before our system shares or receives information.
# Performs website verification, data privacy, wifi checks
import ssl

# Turns code into website and can also be interactable using a web browser
# Storytelling tool for code
# Applications: active web apps, user interactive pages with visualizations, ML codes into applications for demos, quick prototype
import streamlit as st

# Allows computer to generate random values useful in various instances
# Applications: random number selection, games, random password generation
import random

# Focuses on transefering data to ML model understandable format (data in numerical form)
# Better suited for tasks where the focus is on word frequency and importance, rather than understanding the sentiment behind the words.
# Performs: Importance check, rare words boost, comparing text and giving scores based on the topic and feeding ML models these scores so that it will know what are the important words
from sklearn.feature_extraction.text import TfidfVectorizer

# Assigns categories based on the data (fraud detection, customer behavior, binary decisions)
from sklearn.linear_model import LogisticRegression

# ==================================================
# Section 2: Data pre-processing and model training
# ==================================================
ssl._create_default_https_context = ssl._create_unverified_context

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]

vectorizer = TfidfVectorizer()
reg = LogisticRegression(random_state=0, max_iter=10000)

# Data Preparation
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)



# training the model
# Vectorizer will create a matrix of numbers in the transformation process
# Each column is an unique word in the entire patterns array
# Each column is numerical representation of words in each pattern
# Words which are rare will have higher value (IDF) ; frequency of words per sentence is calculated using TF part
# Value obtained after multiplying TF and IDF is the final value in the TF-IDF matrix
# The transformed matrix is compared with the tags and training is done upto 10000 iterations to find the best fit values

x = vectorizer.fit_transform(patterns)
y = tags
reg.fit(x, y)

# ==================================================
# Section 3: Chatbot Function
# ==================================================
def chatbot(input_text):
  ## converting new user inputs into numerical format
    input_text = vectorizer.transform([input_text])
    ## retrieving the first predicted tag as the intent of the user's input
    tag = reg.predict(input_text)[0]
    ## iterates through list of intents to get a random response for the respective tag
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# variable keeping track of the interactions with the chatbot
counter = 0

# ==================================================
# Section 3: Chatbot Conversation Handler
# ==================================================
def main():
    global counter

    ## st - streamline library and its function to build interactive webpage
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1

    # Asks the user to input a message using a text input field.
    # Key is a optional parameter to keep track of the number of interactions with the user
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    # After receiving user input, chatbot function is called
    # The response is displayed in a text area
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

    ## if the user inputs any of the strings like 'goodbye' or 'bye', the chatbot ends the conversation with a thank you message
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()