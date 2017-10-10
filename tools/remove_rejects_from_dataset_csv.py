from check_dataset import remove_rejects_from_dataset_csv
import argparse
import os

def main():
    """Remove rjected CZI files from datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_source', help='path to directory of dataset CSVs')
    opts = parser.parse_args()
    for entry in os.scandir(opts.path_source):
        path_lower = entry.path.lower()
        if path_lower.endswith('_train.csv') or path_lower.endswith('_test.csv'):
            print('***', entry.path, '***')
            remove_rejects_from_dataset_csv(
                path_dataset_csv = entry.path,
                path_rejects_csv = 'data/dataset_eval/czi_rejects.csv',
            )

if __name__ == '__main__':
    main()
