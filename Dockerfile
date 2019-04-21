FROM youyu/ubuntu:16.04

MAINTAINER Uehwan <ohnefetter@kaist.ac.kr>

RUN apt-get update && apt-get install -y libpcl-dev pcl-tools
RUN cd /home && mkdir /slam
