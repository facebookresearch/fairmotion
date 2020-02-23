import argparse
from collections import defaultdict
import os

MOCAP_FORMATS = [".c3d", ".amc", ".bvh", ".fbx", ".v", ".x2d", ".xcp", ".hdf", ".trial"]

def walk_dir(folder, output_file):
    print(folder)
    for root, dirs, files in os.walk(folder, topdown=False):
        for dir in dirs:
            walk_dir(dir, output_file)

        available_formats = defaultdict(list)
        for file in files:
            if any([str.endswith(file.lower(), format) for format in MOCAP_FORMATS]):
                filename = ".".join(file.split(".")[:-1])
                format = file.split(".")[-1]
                available_formats[filename].append(format)

        for file, formats in available_formats.items():
            with open(output_file, "a") as f:
                f.write(f"{os.path.join(root, file)},{','.join(formats)}\n")


def main(args):
    walk_dir(args.root_path, args.output_file)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='List files in disk')
    parser.add_argument("--root-path", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    main(args)
