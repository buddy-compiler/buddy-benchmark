import jenkins.model.*
import hudson.model.*
import org.jenkinsci.plugins.workflow.job.*
import org.jenkinsci.plugins.workflow.cps.*

def jenkins = Jenkins.instance
jenkins.setLabelString("JenkinsNode")

// List of job names and corresponding Jenkinsfile paths
def jobDetails = [
    ['buddy_DeepLearningBenchmark', '/usr/share/jenkins/ref/init.groovy.d/jenkinsfileBuddy'],
    //['opencv_test', '/usr/share/jenkins/ref/init.groovy.d/jenkinsfileOpencv']
]

jobDetails.each { jobName, jenkinsfilePath ->
    def job = jenkins.createProject(WorkflowJob, jobName)
    def pipelineScript = new File(jenkinsfilePath).text

    def flowDefinition = new CpsFlowDefinition(pipelineScript, true)
    job.setDefinition(flowDefinition)

    job.save()

    println("Pipeline job '${jobName}' created successfully.")
}

