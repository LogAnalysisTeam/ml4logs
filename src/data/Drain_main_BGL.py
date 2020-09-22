#!/usr/bin/env python
import sys

# sys.path.append('../')
from logparser import Drain

print("arguments: {}".format(sys.argv))

input_dir = sys.argv[1]  # The input directory of log file
log_file = sys.argv[2]  # 'BGL.log'  # The input log file name
output_dir = sys.argv[3]  # The output directory of parsing results
log_format = '<Label> <Timestamp> <Date> <Id1> <Time> <Id2> <User> <Group> <Level> <Content>'  # BGL log format
# Regular expression list for optional preprocessing (default: [])
regex = [
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
parser.parse(log_file)
