import com.cloudbees.plugins.credentials.CredentialsScope
import com.cloudbees.plugins.credentials.domains.Domain
import com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl
import com.cloudbees.plugins.credentials.SystemCredentialsProvider
import jenkins.model.Jenkins
import hudson.slaves.*
import hudson.plugins.sshslaves.*
import jenkins.*
import jenkins.model.*
import hudson.*
import hudson.model.*

// 加载配置文件
def props = new Properties()
File configFile = new File("/usr/share/jenkins/ref/init.groovy.d/config.properties")
props.load(new FileInputStream(configFile))

// 创建凭据
def createCredentials(String id, String description, String username, String password) {
    def credentials = new UsernamePasswordCredentialsImpl(CredentialsScope.GLOBAL, null, description, username, password)
    SystemCredentialsProvider.getInstance().getStore().addCredentials(Domain.global(), credentials)
    return credentials.getId()
}

// 创建 SSHLauncher
def createSSHLauncher(String host, int port, String credentialsId) {
    return new hudson.plugins.sshslaves.SSHLauncher(
        host,
        port,
        credentialsId,
        '', '', '', '', 60, 5, 15,
        new hudson.plugins.sshslaves.verifiers.NonVerifyingKeyVerificationStrategy()
    )
}

// 创建 Jenkins 节点
def createNode(String nodeName, String remoteFS, String label, SSHLauncher launcher) {
    def slave = new DumbSlave(nodeName, '', remoteFS, '1', Node.Mode.EXCLUSIVE, label, launcher, new RetentionStrategy.Always())
    Jenkins.instance.addNode(slave)
}

// 获取所有节点名
def nodeNames = props.stringPropertyNames().findAll { it.endsWith('.host') }.collect { it.split('\\.')[0] }

// 遍历所有节点并创建
nodeNames.each { nodeName ->
    def host = props.getProperty("${nodeName}.host")
    def port = props.getProperty("${nodeName}.port") as int
    def username = props.getProperty("${nodeName}.username")
    def password = props.getProperty("${nodeName}.password")
    def remoteFS = props.getProperty("${nodeName}.remoteFS")

    // 创建凭据
    def credentialsId = createCredentials(nodeName, "${nodeName} 登录凭据", username, password)

    // 创建 SSHLauncher
    def launcher = createSSHLauncher(host, port, credentialsId)

    // 创建 Jenkins 节点
    createNode(nodeName, remoteFS, nodeName, launcher)

    println "节点 ${nodeName} 创建成功"
}

println "所有节点创建成功"
