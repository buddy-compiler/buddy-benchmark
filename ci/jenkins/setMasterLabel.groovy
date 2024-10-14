import jenkins.model.Jenkins

def builtInNode = Jenkins.instance

builtInNode.setLabelString("JenkinsNode")

println("Label 'JenkinsNode' added to the built-in Jenkins node")
