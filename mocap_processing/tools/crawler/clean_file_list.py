import argparse

# pip install c3d
import c3d

# pip install bvh
from bvh import Bvh
from enum import Enum


class MocapFormat(Enum):
    c3d = 1
    bvh = 2


def filter_file_list(csv_file):
    DOES_NOT_CONTAIN = [
        "$RECYCLE.BIN",
        "Program Files",
        "Seagate Desktop Drive/Adithya_arm_test/",
        "Program Files/Vicon",
        "SnakeBot",
        "L/H/Robots",
    ]
    with open(csv_file) as file, open(
        f"{args.file.split('.')[0]}_clean.csv", "w"
    ) as output_file:
        for line in file:
            file_path = line.split(",")[0]
            # Conditions to clean
            exclude = False
            for string in DOES_NOT_CONTAIN:
                if string in file_path:
                    exclude = True
            if not exclude:
                output_file.write(line)


def modify_filename(filename: str):
    return filename.replace("/Volumes", "/media/dgopinath", 1)


def print_stats(csv_file, mocap_format):
    total_files = 0
    total_time = 0.0
    time = {}
    with open(csv_file) as file:
        for line in file:
            file_path = line.split(",")[0]
            formats = line.split(",")[1:]
            if mocap_format.name not in formats:
                continue
            if mocap_format == mocap_format.c3d:
                time[line] = get_c3d_time(file_path)
            elif mocap_format == MocapFormat.bvh:
                time[line] = get_bvh_time(file_path)
    print(
        f"Total time: {sum(time.values())}\n"
        f"Total files: {len(time)}\n"
        f"Max time: {max(time.items(), key=lambda t: t[1])}"
    )


def get_c3d_time(filename: str):
    try:
        c3d_header = c3d.Header(open(f"{modify_filename(filename)}.c3d", "rb"))
    except Exception as e:
        print(e, filename)
        return 0.0
    frame_rate = c3d_header.frame_rate if c3d_header.frame_rate > 1 else 120.0
    return (c3d_header.last_frame + 1) / frame_rate


def get_bvh_time(filename: str):
    with open(f"{modify_filename(filename)}.bvh") as f:
        try:
            bvh_file = Bvh(f.read())
        except PermissionError as e:
            print(e, filename)
            return 0.0
        time = 0.0
        try:
            time = bvh_file.nframes * bvh_file.frame_time
        except Exception as e:
            print(e, filename)
        return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean files in output of crawler")
    parser.add_argument("--task", type=str)
    parser.add_argument("--file", type=str, required=False)
    args = parser.parse_args()
    if args.task == "clean":
        filter_file_list(args.file)
    if args.task == "c3d_stats":
        print_stats(args.file, MocapFormat.c3d)
    if args.task == "bvh_stats":
        print_stats(args.file, MocapFormat.bvh)
