# Import necessary libraries
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/pardalito/anaconda3/envs/meia/lib
import numpy as np
import tensorflow as tf
import os
# Define the training data
with open('demo.txt', 'r') as file:
    text = file.read()

# Define the vocabulary and the mapping of characters to integers
vocab = sorted(set(text))
char_to_idx = {char:idx for idx, char in enumerate(vocab)}

# Convert the text to numbers
text_as_int = np.array([char_to_idx[char] for char in text])

# Define the maximum sequence length and the batch size
seq_length = 50
batch_size = 64

# Create training examples and targets by sliding a window of size seq_length
examples_per_epoch = len(text) // (seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Define the input and output examples
def create_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Map each sequence to an input and target example
dataset = sequences.map(create_input_target)

# Create the RNN model with LSTM units
num_chars = len(vocab)
rnn_units = 1024
embedding_dim = 256
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_chars, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(num_chars)
])

# Define the loss and optimizer functions
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Define the checkpoint callback to save the model during training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

# Train the model
epochs = 10
history = model.fit(dataset.batch(batch_size), epochs=epochs, callbacks=[checkpoint_callback])

# Use the trained model to generate new text
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Define the temperature parameter for the softmax function
def generate_text(model, start_string):
  num_generate = 1000
  input_eval = [char_to_idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)

      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(vocab[predicted_id])

  return (start_string + ''.join(text_generated))