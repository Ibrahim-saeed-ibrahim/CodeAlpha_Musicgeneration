import pickle
import numpy as np
from keras.models import load_model
from music21 import instrument, note, chord, stream

# Load mappings
with open("mapping.pkl", "rb") as f:
    pitchnames = pickle.load(f)

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

note_to_int = {note: number for number, note in enumerate(pitchnames)}
int_to_note = {number: note for number, note in enumerate(pitchnames)}

n_vocab = len(pitchnames)
seq_len = 100

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = load_model("model.h5")

# ---------------------------------------------------------
# Pick a random starting sequence
# ---------------------------------------------------------
start_index = np.random.randint(0, len(notes) - seq_len)
pattern = [note_to_int[note] for note in notes[start_index:start_index + seq_len]]

generated = []

# ---------------------------------------------------------
# Generate 500 notes
# ---------------------------------------------------------
for _ in range(500):
    x = np.reshape(pattern, (1, seq_len, 1))
    x = x / float(n_vocab)

    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]

    generated.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# ---------------------------------------------------------
# Convert notes to MIDI
# ---------------------------------------------------------
output_notes = []

for pattern in generated:
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        output_notes.append(new_note)

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output.mid')

print("Generated output.mid")
