# 使用 Jenkins 官方基础镜像
FROM jenkins/jenkins:2.462.1-jdk17

# 使用 root 权限来安装插件
USER root

# 安装指定的插件
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli -f /usr/share/jenkins/ref/plugins.txt

# 拷贝创建节点脚本和创建测试任务脚本，Jenkins 初始运行自动执行
COPY --chown=jenkins:jenkins createNode.groovy config.properties /usr/share/jenkins/ref/init.groovy.d/
COPY --chown=jenkins:jenkins jenkinsfile createJob.groovy /usr/share/jenkins/ref/init.groovy.d/
COPY --chown=jenkins:jenkins jenkinsfile setMasterLabel.groovy /usr/share/jenkins/ref/init.groovy.d/

RUN apt-get update && \
    apt-get install -y  --no-install-recommends binutils\
            cmake ninja-build make git  && \
    rm -rf /var/lib/apt/lists/*

# 在首次启动时跳过默认的安装引导页面
ENV JAVA_OPTS=-Djenkins.install.runSetupWizard=false

# 切换回 Jenkins 用户
USER jenkins
