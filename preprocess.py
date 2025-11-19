from music21 import converter, instrument, note, chord
import glob
import pickle
import glob
import os

# Search recursively for both .mid and .midi files
midi_files = glob.glob(os.path.join("dataset", "**", "*.mid"), recursive=True)
midi_files += glob.glob(os.path.join("dataset", "**", "*.midi"), recursive=True)

print("Found MIDI files:", midi_files)


notes = []

for file in glob.glob("dataset/**/*.mid", recursive=True):
    print("Parsing:", file)
    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Saved notes.pkl with", len(notes), "notes")
