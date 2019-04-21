FROM youyu/ubuntu:16.04

MAINTAINER Uehwan <ohnefetter@kaist.ac.kr>

RUN apt-get install -y libpcl-dev pcl-tools
RUN apt-get install -y snapd \
&& snap install -y code --classic
