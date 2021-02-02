FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip

COPY . /opt/CommunityQ/
WORKDIR /opt/CommunityQ/
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 api.py