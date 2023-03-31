import tensorflow as tf
import numpy as np

# load the text corpus
with open('./NLP/demo.txt', 'r') as file:
    corpus = file.read()

# tokenize the text corpus into words
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([corpus])
word_to_index = tokenizer.word_index
index_to_word = dict((i, w) for w, i in word_to_index.items())

# create sequences of fixed length with a sliding window
seq_length = 20
step_size = 1
sequences = []
next_words = []
words = corpus.split()
vocab_set = set(word_to_index.keys())
for i in range(0, len(words) - seq_length - 1, step_size):
    sequence = ' '.join(words[i:i+seq_length])
    next_word = words[i+seq_length]
    if next_word in vocab_set:
        sequences.append(sequence)
        next_words.append(next_word)

# convert sequences to numerical data
X = tokenizer.texts_to_sequences(sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=seq_length)
categorical = tokenizer.texts_to_sequences(next_words)
categorical_flat = np.array(categorical).flatten()
y = tf.keras.utils.to_categorical(categorical_flat, num_classes=len(word_to_index))

# define the LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(word_to_index) + 1, output_dim=128, input_length=seq_length))
model.add(tf.keras.layers.LSTM(units=128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=len(word_to_index), activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train the model
model.fit(X, y, epochs=50, batch_size=128)

# generate text
seed_text = 'The quick brown fox'
generated_text = seed_text.lower()
for i in range(100):
    x = tokenizer.texts_to_sequences([generated_text])[0]
    x = tf.keras.preprocessing.sequence.pad_sequences([x], maxlen=seq_length)
    prediction = model.predict(x, verbose=0)[0]
    index = np.argmax(prediction)
    word = index_to_word[index]
    generated_text += ' ' + word

print(generated_text)
