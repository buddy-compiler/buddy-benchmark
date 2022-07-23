# skip first 3 line.
for i in range(3):
    input()
while True:
    try:
        line = input()
        name, time, _, time_cpu, _, it_times = line.split()

        time_cpu = float(time_cpu)
        it_times = int(it_times)
        
        gflops = (2.0 * it_times * 2088 * 2048 * 2048 ) / time_cpu
        print(gflops, 'GFLOPS')
    except EOFError:
        break
    
