#!/bin/bash

DDIR=`readlink -f ../data/raw`
mkdir $DDIR/BGL
mkdir $DDIR/HDFS1
mkdir $DDIR/HDFS2

# BGL
cd $DDIR/BGL
wget -O BGL.tar.gz https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1
tar xvzf BGL.tar.gz
rm BGL.tar.gz

#HDFS

cd $DDIR/HDFS1
wget -O HDFS_1.tar.gz https://zenodo.org/record/3227177/files/HDFS_1.tar.gz?download=1
tar xvzf HDFS_1.tar.gz
rm HDFS_1.tar.gz

cd $DDIR/HDFS2
wget -O HDFS_2.tar.gz https://zenodo.org/record/3227177/files/HDFS_2.tar.gz?download=1
tar xvzf HDFS_2.tar.gz
rm HDFS_2.tar.gz
