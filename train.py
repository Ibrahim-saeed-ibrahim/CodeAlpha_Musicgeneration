import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

# ---------------------------------------------------------
# Load notes
# ---------------------------------------------------------
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Create vocab
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

# Map notes to integers
note_to_int = {note: number for number, note in enumerate(pitchnames)}

# ---------------------------------------------------------
# Prepare sequences
# ---------------------------------------------------------
seq_len = 100
network_input = []
network_output = []

for i in range(0, len(notes) - seq_len):
    seq_in = notes[i:i + seq_len]
    seq_out = notes[i + seq_len]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, seq_len, 1))
network_input = network_input / float(n_vocab)
network_output = to_categorical(network_output)

# ---------------------------------------------------------
# Build LSTM model
# ---------------------------------------------------------
model = Sequential([
    LSTM(512, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.3),
    LSTM(512, return_sequences=True),
    Dropout(0.3),
    LSTM(512),
    Dense(256, activation='relu'),
    Dense(n_vocab, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
model.fit(network_input, network_output, epochs=50, batch_size=64)
model.save("model.h5")

with open("mapping.pkl", "wb") as f:
    pickle.dump(pitchnames, f)

print("Training completed.")
