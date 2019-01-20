import pickle

with open('./plays_data.pickle', 'rb') as f:
    plays_data = pickle.load(f)

for play in plays_data:
    print(plays_data[play]['nouns'])