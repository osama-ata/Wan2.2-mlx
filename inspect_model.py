import argparse
from safetensors import safe_open

def inspect_model(model_path):
    with safe_open(model_path, framework="mlx") as f:
        for key in f.keys():
            print(key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a safetensors model file.")
    parser.add_argument("model_path", type=str, help="The path to the model file.")
    args = parser.parse_args()
    inspect_model(args.model_path)
