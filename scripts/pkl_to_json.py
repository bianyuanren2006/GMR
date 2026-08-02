"""
Convert GMR pickle file to JSON format.
"""
import argparse
import json
import pickle
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def pkl_to_json(pkl_path, json_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Convert numpy arrays to lists for JSON serialization
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = v.tolist()
        elif isinstance(v, np.integer):
            output[k] = int(v)
        elif isinstance(v, np.floating):
            output[k] = float(v)
        elif isinstance(v, list):
            # Check if items are numpy arrays
            if v and isinstance(v[0], np.ndarray):
                output[k] = [x.tolist() for x in v]
            else:
                output[k] = v
        else:
            output[k] = v

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"Converted {pkl_path} → {json_path}")
    print(f"  fps: {output.get('fps', '?')}")
    print(f"  frames: {len(output.get('root_pos', []))}")
    print(f"  dof_names: {len(output.get('dof_pos', [[]])[0])} joints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pkl to JSON")
    parser.add_argument("--input", required=True, help="Input .pkl file")
    parser.add_argument("--output", required=True, help="Output .json file")
    args = parser.parse_args()
    pkl_to_json(args.input, args.output)
