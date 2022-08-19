# skip first 3 line.
for i in range(3):
    input()
while True:
    try:
        line = input()
        name, time, _, time_cpu, _, it_times = line.split()
        name, base_size = name.split('/')

        time_cpu = float(time_cpu)
        it_times = int(it_times)
        base_size = int(base_size)
        
        gflops = (2.0 * it_times * base_size * base_size * base_size) / time_cpu
        print(f'{name}\t\t {base_size}\t\t {gflops}')
    except EOFError:
        break
    
