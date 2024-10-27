FROM jenkins/jenkins:2.462.1-jdk17

USER root

# Install the plugins used in Jenkins
COPY --chown=jenkins:jenkins plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli -f /usr/share/jenkins/ref/plugins.txt

# Copy the node creation script and the test job creation script, which are automatically executed by the initial run of Jenkins
COPY --chown=jenkins:jenkins createNode.groovy config.properties* /usr/share/jenkins/ref/init.groovy.d/
COPY --chown=jenkins:jenkins jenkinsfileBuddy jenkinsfileOpencv createJob.groovy /usr/share/jenkins/ref/init.groovy.d/

# Install dependencies for test tasks
COPY requirementsPy.txt  requirementsApt.txt /app/
RUN apt-get update && \
    xargs -a /app/requirementsApt.txt apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* 
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r /app/requirementsPy.txt
ENV PATH="/opt/venv/bin:$PATH"

# Skip the default installation boot page on first startup
ENV JAVA_OPTS=-Djenkins.install.runSetupWizard=false

USER jenkins
