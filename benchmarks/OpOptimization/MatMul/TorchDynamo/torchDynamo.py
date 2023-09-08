import torch


def matrix_multiply(matrix1, matrix2):
    m, n = matrix1.size()
    n, p = matrix2.size()
    result = torch.zeros(m, p)
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result


def default_matrix_multiply():
    def inner_matrix_multiply(matrix1, matrix2):
        m, n = matrix1.size()
        n, p = matrix2.size()
        result = torch.zeros(m, p)
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        
        return result
    return inner_matrix_multiply




def dynamo_matrix_multiply():
    # compiled_mm = torch.compile(matrix_multiply, backend="inductor")
    compiled_mm = torch.compile(matrix_multiply,mode="max-autotune")
    return compiled_mm







