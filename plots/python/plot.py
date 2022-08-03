#!/usr/bin/python3
import argparse
from plotools.compare import compare_wave


def parse_args():
    parser = argparse.ArgumentParser(description="This module is designed for figure plotting.")
    parser.add_argument('file1', help='The path of the .wav file before process.')
    parser.add_argument('file2', help='The path of the .wav file after process.')
    parser.add_argument('-b', '--part', choices=['head', 'body', 'tail', 'all'], default='body',
                        help='Choose a part of the time-domain figure to show. Possible values are "head", "body", '
                             '"tail" and "all".')
    parser.add_argument('-p', '--portion', type=float,
                        help='Decide the portion(between 0 and 1) of the wave to be shown. Note once this argument '
                             'was specified, the "--points" argument would be invalid.')
    parser.add_argument('-n', '--points', type=int, default=5000,
                        help='Decide the number of points of the wave to be shown.')
    parser.add_argument('--nosave', action='store_false', help='Decide whether or not to save the figure.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Plotting now...")
    compare_wave(args.file1, args.file2, part=args.part, 
                 granularity=args.portion if args.portion else args.points,
                 save=args.nosave)
    print("Plotting done!")
