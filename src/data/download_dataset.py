# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from os.path import join as pjoin
import requests
import tarfile

known_datasets = {
    'BGL': {'url': 'https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1', 'file': 'BGL.tar.gz'},
    'HDFS1': {'url': 'https://zenodo.org/record/3227177/files/HDFS_1.tar.gz?download=1', 'file': 'HDFS_1.tar.gz'},
    'HDFS2': {'url': 'https://zenodo.org/record/3227177/files/HDFS_2.tar.gz?download=1', 'file': 'HDFS_2.tar.gz'},
}

@click.command()
@click.argument('datasets', type=click.Choice(known_datasets, case_sensitive=False), nargs=-1)
def main(datasets):
    """ Downloads raw dataset(s)
    """
    logger = logging.getLogger(__name__)
    
    if len(datasets) == 0:
        datasets = sorted(list(known_datasets.keys()))

    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = pjoin(project_dir, 'data', 'raw')
    logger.info(project_dir)
    for d in datasets:
        rdir = pjoin(raw_dir, d)
        os.makedirs(rdir, exist_ok=True)
        logger.info('downloading {} dataset to {}'.format(d, rdir))
        response = requests.get(known_datasets[d]['url'])
        outfile = pjoin(rdir, known_datasets[d]['file'])
        filename = Path(outfile)
        filename.write_bytes(response.content)
        logger.info('extracting archive: {}'.format(outfile))
        tar = tarfile.open(outfile, "r:gz")
        tar.extractall(rdir)
        tar.close()
        logger.info('deleting archive: {}'.format(outfile))
        os.remove(outfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
