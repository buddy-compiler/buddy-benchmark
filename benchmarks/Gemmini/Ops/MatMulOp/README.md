# Exo dynamic

## 步骤
1. 需要exo依赖 见 [exo github](https://github.com/exo-lang/exo)
2. 这里使用exo并且使用exocc将python文件转化为C Kernel，然后将ExoMatmul.c中Kernel部分的函数替换为exocc编译后的结果。 
```shell
exocc matmul_dynamic.py
```
3. 在matmul_dynamic中有两个Kernel：schedule_matmul_4 & schedule_matmul_flexible
schedule_matmul_4 可以通过部分shape，但是具有较好的优化效果，通过修改MM,NN,KK的数值来修改shape
schedule_matmul_flexible 可以通过所有的规则shape(至少16的整数倍)，但是性能不佳。通过给函数传入不同的参数来修改shape
4. test_flexible_kernels.py用于测试schedule_matmul_flexible的生成结果 涵盖了大部分类型的shape测试，均通过编译。