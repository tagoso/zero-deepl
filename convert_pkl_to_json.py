import pickle
import json
import numpy as np

with open("sample_weight.pkl", "rb") as f:
    data = pickle.load(f)

# Convert NumPy arrays to lists so they can be written to JSON
json_ready = {k: v.tolist() for k, v in data.items()}

with open("sample_weight.json", "w") as f:
    json.dump(json_ready, f)
