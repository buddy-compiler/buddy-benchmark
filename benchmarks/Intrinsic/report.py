import json
import argparse
import os
import re
import sys
import csv

def load_benchmark_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None

def normalize_test_name(name):
    name = re.sub(r'/iterations:\d+', '', name)
    name = re.sub(r'/repeats:\d+', '', name)
    if name.endswith('/'):
        name = name[:-1]
    if name.endswith('_mean'):
        name = name[:-5]

    return name

def remove_common_prefix(filenames):
    if not filenames:
        return filenames
    base_names = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    prefix = os.path.commonprefix(base_names)
    if len(prefix) > 3:
        return [name[len(prefix):] for name in base_names]
    return base_names

def main():
    parser = argparse.ArgumentParser(description='Compare Google Benchmark JSON results')
    parser.add_argument('json_files', nargs='+', help='JSON benchmark files to compare')
    parser.add_argument('--speedup-only', action='store_true',
                        help='Only show speedup ratios (omit time values)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save CSV output to specified file')
    args = parser.parse_args()

    if len(args.json_files) < 1:
        print("Error: At least one JSON file is required", file=sys.stderr)
        return
    display_names = remove_common_prefix(args.json_files)
    file_data = []
    for i, file_path in enumerate(args.json_files):
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}", file=sys.stderr)
            return

        data = load_benchmark_data(file_path)
        if not data:
            return

        mean_times = parse_mean_times(data)

        if not mean_times:
            print(f"Error: No benchmark results found in {file_path}", file=sys.stderr)
            return

        file_data.append({
            'path': file_path,
            'display_name': display_names[i],
            'means': mean_times
        })
    all_cases = set()
    for data in file_data:
        all_cases.update(data['means'].keys())
    headers = ["Test Case"]
    if not args.speedup_only or len(file_data) == 1:
        for data in file_data:
            headers.append(data['display_name'])
    if len(file_data) > 1:
        for i in range(1, len(file_data)):
            headers.append(f"{file_data[i]['display_name']}/Speedup")
    csv_rows = []
    header_line = "\t".join(headers)
    max_length = min(max(len(header_line), 40), 120)
    print(header_line)
    print('-' * max_length)
    for case in sorted(all_cases):
        row = [case]
        times = []
        for data in file_data:
            time_val = data['means'].get(case)
            times.append(time_val)
        if not args.speedup_only or len(times) == 1:
            for t in times:
                value_str = f"{t:.4f}" if t is not None else "N/A"
                row.append(value_str)
        if len(times) > 1:
            base_time = times[0]
            if base_time is None or base_time == 0:
                for i in range(1, len(times)):
                    row.append("N/A")
            else:
                for i in range(1, len(times)):
                    if times[i] is not None and times[i] != 0:
                        speedup = base_time / times[i]
                        value_str = f"{speedup:.4f}"
                        row.append(value_str)
                    else:
                        row.append("N/A")
        print("\t".join(row))
        csv_rows.append(row)
    if args.output:
        try:
            with open(args.output, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                for row in csv_rows:
                    writer.writerow(row)

            print(f"\nCSV results saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving CSV to {args.output}: {str(e)}", file=sys.stderr)

def parse_mean_times(data):
    means = {}
    for item in data.get('benchmarks', []):
        if (item.get('run_type') == 'aggregate' and
            item.get('aggregate_name') == 'mean'):
            normalized_name = normalize_test_name(item['name'])
            means[normalized_name] = item['cpu_time']
    return means

if __name__ == "__main__":
    main()