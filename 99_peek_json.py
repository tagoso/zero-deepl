import json

def analyze_json_structure(data, depth=0, stats=None):
    if stats is None:
        stats = {
            "objects": 0,
            "arrays": 0,
            "strings": 0,
            "numbers": 0,
            "booleans": 0,
            "null": 0,
            "keys": 0,
            "true": 0,
            "false": 0,
            "max_depth": 0
        }

    stats["max_depth"] = max(stats["max_depth"], depth)

    if isinstance(data, dict):
        stats["objects"] += 1
        stats["keys"] += len(data)
        for v in data.values():
            analyze_json_structure(v, depth + 1, stats)
    elif isinstance(data, list):
        stats["arrays"] += 1
        for item in data:
            analyze_json_structure(item, depth + 1, stats)
    elif isinstance(data, str):
        stats["strings"] += 1
    elif isinstance(data, (int, float)):
        stats["numbers"] += 1
    elif isinstance(data, bool):
        stats["booleans"] += 1
        if data:
            stats["true"] += 1
        else:
            stats["false"] += 1
    elif data is None:
        stats["null"] += 1

    return stats

def print_json_summary(stats, data):
    print("General JSON Info:")
    print("------------------")
    print(f" Type:           {type(data).__name__}")
    print(f" Depth:          {stats['max_depth']}")

    print("\nNumber of Data Types:")
    print("---------------------")
    print(f" Number of objects:  {stats['objects']}")
    print(f" Number of arrays:   {stats['arrays']}")
    print(f" Number of strings:  {stats['strings']}")
    print(f" Number of numbers:  {stats['numbers']}")
    print(f" Number of booleans: {stats['booleans']}")
    print(f" Number of null:     {stats['null']}")
    print(f" Number of keys:     {stats['keys']}")
    print(f" Number of true:     {stats['true']}")
    print(f" Number of false:    {stats['false']}")

    if isinstance(data, dict):
        print("\nTop-level Keys:")
        print("---------------")
        for k in data.keys():
            print(f' - "{k}" ({type(data[k]).__name__})')

# -------------------------------
with open("sample_weight.json", "r") as f:
    data = json.load(f)

stats = analyze_json_structure(data)
print_json_summary(stats, data)
