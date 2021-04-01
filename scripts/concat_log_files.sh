#!/bin/bash

cd ../data/raw/Hadoop

mkdir ../../interim/Hadoop

for i in */*.log; do cat $i >> ../../interim/Hadoop/concatenated_hadoop.log;done