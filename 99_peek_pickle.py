import pickle

with open("sample_weight.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
# i.e. <class 'dict'>

for k, v in data.items():
    print(k, v.shape)

print(data["W1"][:5, :5])
