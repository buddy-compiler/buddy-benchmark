import jenkins.model.*
import hudson.model.*
import org.jenkinsci.plugins.workflow.job.*
import org.jenkinsci.plugins.workflow.cps.*

// 获取 Jenkins 实例
def jenkins = Jenkins.instance

def job = jenkins.createProject(WorkflowJob, 'opencv_test')

// 设置流水线定义为读取 Jenkinsfile 文件
def pipelineScript = new File("/usr/share/jenkins/ref/init.groovy.d/jenkinsfile").text
def flowDefinition = new CpsFlowDefinition(pipelineScript, true)
job.setDefinition(flowDefinition)

// 保存任务
job.save()

println("Pipeline job 'opencv_test' created successfully.")