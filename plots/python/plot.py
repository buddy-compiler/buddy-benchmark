# ===- plot.py ---------------------------------------------------------===

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===--------------------------------------------------------------------===

# This is the starter of the figure-plotting modules.

# ===--------------------------------------------------------------------===

# !/usr/bin/python3
import argparse
from plotools.compare import compare_wave


def parse_args():
    parser = argparse.ArgumentParser(
        prog='audio-plot',
        description="This module is designed for figure plotting."
        )
    parser.add_argument(
        'file1', help='The path of the .wav file before process.'
        )
    parser.add_argument(
        'file2', nargs='?', default=None,
        help='The path of the .wav file after process. If no file2 was \
              specified, we will only plot the wave of file1.'
        )
    parser.add_argument('-b', '--part',
                        choices=['head', 'body', 'tail', 'all'],
                        default='body',
                        help='Choose a part of the sound wave in the time-\
                              domain to be plotted. Possible values are \
                              "head", "body", "tail" and "all". If "head" is \
                              specified, the first few points of the sound \
                              wave will be plotted, and "last" stands for \
                              the last few points, while "body" stands for the\
                               center few points. When "all" is selected, the \
                              whole sound wave will be plotted, and the "-p" \
                              and "-n" options will be ignored.')
    parser.add_argument('-p', '--portion', type=float,
                        help='Decide the portion(between 0 and 1) of the wave \
                              to be shown. Note once this argument was \
                              specified, the "--points" argument would be \
                              invalid.')
    parser.add_argument('-n', '--points', type=int, default=5000,
                        help='Decide the number of points of the sound wave to\
                               be plotted. The default number is 5000, which \
                              is cosy enough to show the wave.')
    parser.add_argument('--nosave', action='store_false',
                        help='Decide whether to save the figure. If this \
                              option is set, the result figure would only be \
                              shown on the screen instead of being saved as a \
                              file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Plotting now...")
    compare_wave(args.file1, args.file2, part=args.part,
                 granularity=args.portion if args.portion else args.points,
                 save=args.nosave)
    print("Plotting done!")
