import json
import sys
from pathlib import Path

def pretty_json(input_file, output_file=None):

    input_path = Path(input_file)

    if output_file is None:
        output_file = input_path.stem + "_pretty.json"

    with open(input_file, "r") as f:
        data = json.load(f)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Formatted JSON saved to: {output_file}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python pretty_json.py <input_json> [output_json]")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        output_file = None

    pretty_json(input_file, output_file)
