# 测试平台搭建

构建平台：

```bash
docker-compose up
```



命令运行后，会得到 Jenkins、llvm-17、llvm-18、riscv-gcc-13、riscv-gcc-14 这四个镜像。同时，编译工具会挂载到 `./ci/opt` 下，Jenkins 容器启动。通过 8080 端口可以访问 Jenkins web 页面。Jenkins 中应包括配置完成的工作节点，以及一个初始测试项目 opencv_test

