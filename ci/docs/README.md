# 测试平台搭建

构建平台：

```bash
docker-compose up
```



命令运行后，会得到 Jenkins、llvm-17、llvm-18、riscv-13、riscv-14 这四个镜像。编译工具会以数据卷的形式保存到宿主机下，Jenkins 容器启动。通过 8080 端口可以访问 Jenkins web 页面

