import os
import csv
import argparse
import subprocess


def get_file_hash(path: str):
    file_hash = []
    for root, _, files in os.walk(path):
        for file in files:
            sha256hash = subprocess.run(
                ['sha256sum', os.path.join(root, file)],
                capture_output=True, text=True
            )
            file_hash.append([file, str(sha256hash.stdout[:64])])
    return file_hash


def write_csv(file_hash: list, out_file: str):
    field = ['filename', 'hash']
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(field)
        writer.writerows(file_hash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get filehash')
    parser.add_argument('--path', default='../_dataset/processed_sprite')
    parser.add_argument('--out_file', default='hash_sprite.csv')
    args = parser.parse_args()

    file_hash = get_file_hash(args.path)
    write_csv(file_hash, args.out_file)
