import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext, Frame

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:/Users/JEFFREY/Desktop/ALL IN HERE/portfolio proj/chat bot/intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]  # Suppress Keras verbose output
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    user_message = user_entry.get()
    if user_message.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_message + '\n\n')
        chat_window.config(foreground="#442265", font=("Arial", 12))
        user_entry.delete(0, tk.END)

        ints = predict_class(user_message)
        res = get_response(ints, intents)
        chat_window.insert(tk.END, "Bot: " + res + '\n\n')
        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

# GUI setup
root = tk.Tk()
root.title("Chatbot")
root.geometry("500x600")
root.resizable(width=False, height=False)

# Frame for chat window
chat_frame = Frame(root)
chat_frame.pack(pady=10)

chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=60, height=20, font=("Arial", 12), bd=0, bg="#f4f4f4", fg="#000", state=tk.DISABLED)
chat_window.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

# Frame for user input
input_frame = Frame(root)
input_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

user_entry = tk.Entry(input_frame, font=("Arial Narrow", 14), width=38, bd=1, relief=tk.SOLID)
user_entry.pack(side=tk.LEFT, padx=(10, 5), pady=(0, 10))

send_button = tk.Button(input_frame, text="Send", font=("Arial", 12, "bold"), bg="#442265", fg="#fff", command=send_message)
send_button.pack(side=tk.LEFT, padx=(5, 10), pady=(0, 10))

root.bind('<Return>', send_message)

root.mainloop()
