import os
import shutil
import argparse

def delete_dirs(parent_dir, substring):
    for root, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            if substring in dir:
                shutil.rmtree(os.path.join(root, dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir', type=str, default='exp')
    parser.add_argument('--substring', type=str, default='sampleNone')
    args = parser.parse_args()

    delete_dirs(args.parent_dir, args.substring)

        