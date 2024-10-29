import numpy as np
import sys
import os
import timeit

if len(sys.argv) != 2:
    print("need file path")
    sys.exit(1)

output_path = sys.argv[1]
output_file = os.path.join(output_path, 'result.txt')
time_output_file = os.path.join(output_path, 'numpyTimeResult.txt')

if os.path.exists(output_file):
    os.remove(output_file)
if os.path.exists(time_output_file):
    os.remove(time_output_file)

def compute_rfft():
    arr = np.arange(0, 840)
    rfft_result = np.fft.rfft(arr)
    output_lines = []
    first_value = rfft_result[0].real
    output_lines.append(f"{first_value:.6f}")
    for c in rfft_result[1:-1]:
        real_part = c.real
        imag_part = c.imag
        output_lines.append(f"{real_part:.6f}")
        output_lines.append(f"{imag_part:.6f}")
    last_value = rfft_result[-1].real
    output_lines.append(f"{last_value:.6f}")
    return output_lines

# set a timer with timeit
execution_time = timeit.timeit(compute_rfft, number=1) * 1000

# get RFFT result
rfft_result_lines = compute_rfft()

# record the result
with open(output_file, 'w') as f:
    for line in rfft_result_lines:
        f.write(line + '\n')

with open(time_output_file, 'w') as f_time:
    f_time.write(f"Execution time for RFFT: {execution_time:.6f} milliseconds\n")


print(f"RFFT result saved to '{output_file}'")
print(f"Execution time saved to '{time_output_file}'")
