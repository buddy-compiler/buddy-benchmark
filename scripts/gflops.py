def calc_conv(p):
    '''
  long a = 1, b = factor, c = 16 * factor, d = 16 * factor,
       e = 1, f = 32 * factor, g = 32 * factor;
    '''

    a = 1
    b = p
    c = 16 * p
    d = 16 * p
    e = 1
    f = 32 * p
    g = 32 * p

    res = 2.0 * a * b * c * d * e * f * g 
    # print(a, b, c, d, e, f, g, res)
    return res

def calc_gemm(p):
    return 2.0 * p * p * p

calc_dict = {
        'BM_CONV': calc_conv,

        'BM_GEMM': calc_gemm,
        'BM_OPENCV_GEMM': calc_gemm
    }
    
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
        
        gflops = (1.0 * it_times * calc_dict[name](base_size)) / time_cpu
        print(f'{name}\t\t {base_size}\t\t {gflops}')
    except EOFError:
        break
    
