import argparse

def main(args):
    DOES_NOT_CONTAIN = ["$RECYCLE.BIN", "Program Files"]
    with open(args.file) as csv_file, open(f"{args.file.split('.')[0]}_clean.csv", "w") as output_file:
        for line in csv_file:
            file_path = line.split(",")[0]
            # Conditions to clean
            exclude = False
            for string in DOES_NOT_CONTAIN:
                if string in file_path:
                    exclude = True
            if not exclude:
                output_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean files in output of crawler')
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    main(args)
