import pickle


with open('dog.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)