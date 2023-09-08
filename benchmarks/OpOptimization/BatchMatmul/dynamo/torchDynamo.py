import torch


def batch_matrix_multiply(tensor1, tensor2):
    # 获取输入矩阵的维度
    b, m, n = tensor1.size()
    b, n, p = tensor2.size()
    
    # 创建一个全零的结果矩阵，维度为 m x p
    result = torch.zeros(b, m, p)
    
    # 进行批量矩阵乘法
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
        # 获取输入矩阵的维度
        b, m, n = tensor1.size()
        b, n, p = tensor2.size()
        
        # 创建一个全零的结果矩阵，维度为 m x p
        result = torch.zeros(b, m, p)
        
        # 进行批量矩阵乘法
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
    # compiled_mm = torch.compile(matrix_multiply, backend="inductor")
    compiled_mm = torch.compile(batch_matrix_multiply,mode="max-autotune")
    return compiled_mm







