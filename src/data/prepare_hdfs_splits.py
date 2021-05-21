#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/
#
# Modified for the purposes of Contextual Embeddings for Anomaly Detection in Logs thesis
#
#

import argparse

from hdfs_korytak import prepare_and_save_splits


def custom_int_type(x: str) -> int:
    tmp = int(x)
    if tmp < 1:
        raise argparse.ArgumentTypeError('This value cannot be less than 1.')
    return tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process HDFS1 data set and split data into training, validation and '
                                                 'testing sets.')
    parser.add_argument('n_folds', action='store', type=custom_int_type, help='a number of cross validation splits')
    parser.add_argument('-in', type=str, metavar='PATH/TO/FOLDER', dest='input', default='../../data/raw/HDFS1',
                        help='a location with HDFS1 data (HDFS.log and anomaly_label.csv) '
                             '(default: ../../data/raw/HDFS1)')
    parser.add_argument('-out', type=str, metavar='PATH/TO/FOLDER', dest='output', default='../../data/interim/HDFS1',
                        help='a location where all intermediate data will be saved (default: ../../data/interim/HDFS1)')

    args = parser.parse_args()

    prepare_and_save_splits(args.input, args.output, args.n_folds)
