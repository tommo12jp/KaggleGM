FROM ubuntu

RUN apt-get update
RUN apt-get install -y sudo
RUN sudo apt-get update
RUN sudo apt install -y python3-pip
#RUN sudo apt install -y nodejs npm
RUN sudo apt install -y language-pack-ja
RUN sudo update-locale LANG=ja_JP.UTF-8
RUN pip3 install jupyterlab
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install sklearn
RUN pip3 install seaborn
RUN pip3 install argparse
RUN pip3 install xgboost
RUN apt-get install -y git 