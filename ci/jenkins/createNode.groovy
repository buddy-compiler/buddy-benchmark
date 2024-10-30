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

// Load the configuration file
def props = new Properties()
File configFile = new File("/usr/share/jenkins/ref/init.groovy.d/config.properties")
props.load(new FileInputStream(configFile))

// Create Credentials
def createCredentials(String id, String description, String username, String password) {
    def credentials = new UsernamePasswordCredentialsImpl(CredentialsScope.GLOBAL, null, description, username, password)
    SystemCredentialsProvider.getInstance().getStore().addCredentials(Domain.global(), credentials)
    return credentials.getId()
}

// Create SSHLauncher
def createSSHLauncher(String host, int port, String credentialsId) {
    return new hudson.plugins.sshslaves.SSHLauncher(
        host,
        port,
        credentialsId,
        '', '', '', '', 60, 5, 15,
        new hudson.plugins.sshslaves.verifiers.NonVerifyingKeyVerificationStrategy()
    )
}

// Create Jenkins slave node
def createNode(String nodeName, String remoteFS, String label, SSHLauncher launcher) {
    def slave = new DumbSlave(nodeName, '', remoteFS, '1', Node.Mode.EXCLUSIVE, label, launcher, new RetentionStrategy.Always())
    Jenkins.instance.addNode(slave)
}

// Get all node names
def nodeNames = props.stringPropertyNames().findAll { it.endsWith('.host') }.collect { it.split('\\.')[0] }

// Iterate over all nodes and create
nodeNames.each { nodeName ->
    def host = props.getProperty("${nodeName}.host")
    def port = props.getProperty("${nodeName}.port") as int
    def username = props.getProperty("${nodeName}.username")
    def password = props.getProperty("${nodeName}.password")
    def remoteFS = props.getProperty("${nodeName}.remoteFS")

    def credentialsId = createCredentials(nodeName, "${nodeName} 登录凭据", username, password)

    def launcher = createSSHLauncher(host, port, credentialsId)

    createNode(nodeName, remoteFS, nodeName, launcher)

    println "Node ${nodeName} created successfully."
}

println "All nodes created successfully"
