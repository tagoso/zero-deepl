import json
import numpy as np

def init_network():
    with open("sample_weight.json", 'r') as f:
        data = json.load(f)

    # list => NumPy Array
    network = {k: np.array(v) for k, v in data.items()}
    return network

network = init_network()
print(network["W1"].shape)
