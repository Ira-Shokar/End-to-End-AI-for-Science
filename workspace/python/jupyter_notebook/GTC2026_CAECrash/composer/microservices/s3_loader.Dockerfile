FROM amazon/aws-cli:latest
RUN yum install -y gzip && yum clean all