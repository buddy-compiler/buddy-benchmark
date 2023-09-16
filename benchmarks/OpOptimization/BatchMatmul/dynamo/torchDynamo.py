import torch


def batch_matrix_multiply(tensor1, tensor2):
    b, m, n = tensor1.size()
    b, n, p = tensor2.size()
    
    result = torch.zeros(b, m, p)

    for batch in range(b):
        matrix1 = tensor1[batch]
        matrix2 = tensor2[batch]
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result


def default_matrix_multiply():
    def inner_matrix_multiply(tensor1, tensor2):
        b, m, n = tensor1.size()
        b, n, p = tensor2.size()
        
        result = torch.zeros(b, m, p)
        
        for batch in range(b):
            matrix1 = tensor1[batch]
            matrix2 = tensor2[batch]
            for i in range(m):
                for j in range(p):
                    for k in range(n):
                        result[i][j] += matrix1[i][k] * matrix2[k][j]
        
        return result
    return inner_matrix_multiply




def dynamo_batch_matrix_multiply():
    compiled_mm = torch.compile(batch_matrix_multiply,mode="max-autotune")
    return compiled_mm







