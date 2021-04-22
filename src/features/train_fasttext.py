import fasttext
import time
from pathlib import Path

def train_fasttext_model(config):
    assert config.output_dir is not None
    assert config.dataset is not None

    ds_path = Path(config.dataset)
    assert ds_path.exists()
    assert ds_path.is_file()

    output_file = Path(config.output_dir) / f'fasttext_from_{ds_path.stem}.bin'

    model = fasttext.train_unsupervised(input=str(ds_path),
                                        model='skipgram',
                                        dim=100,
                                        minn=3,
                                        maxn=6,
                                        minCount=10000,
                                        thread=1)

    model.save_model(str(output_file))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train fasttext model")
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str, help="Dataset to train from")

    config = parser.parse_args()
    train_fasttext_model(config)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')