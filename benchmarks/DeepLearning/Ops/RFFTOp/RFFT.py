import numpy as np
import sys
import os

if len(sys.argv) != 2:
    print("need file path")
    sys.exit(1)

output_path = sys.argv[1]
output_file = os.path.join(output_path, 'result.txt')

arr = np.arange(0, 20)

rfft_result = np.fft.rfft(arr)

output_lines = []

first_value = rfft_result[0].real
output_lines.append(f"{first_value:.0f}")

for c in rfft_result[1:]:
    real_part = c.real
    imag_part = c.imag
    output_lines.append(f"{real_part:.0f}")
    output_lines.append(f"{imag_part:.2f}")

with open(output_file, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"rfft result saved to '{output_file}'")
