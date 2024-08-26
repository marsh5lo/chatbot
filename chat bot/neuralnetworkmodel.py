import random
import json
import os
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD  # Import legacy SGD optimizer

# Correct the variable name to instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open('C:/Users/JEFFREY/Desktop/ALL IN HERE/portfolio proj/chat bot/intents.json').read())

if os.path.exists('bot.json'):  # Correct indentation
    print("The file exists.")
else:
    print("The file does not exist.")

words = []
classes = []
documents = []  # Correct the variable name here
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:  # Corrected the loop variable name
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Change append to extend to add all words
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Correctly pickle dump classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create the bag of words array
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert the training data to a NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Convert the data into two separate arrays
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  # Changed the second Dense layer to 64 units
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Use the legacy SGD optimizer with decay
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print("Done")

