#!/bin/bash

apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
pip install dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $DAGSHUB_UNAME
dvc remote modify origin --local password $DAGSHUB_TOKEN
dvc pull -r origin

apt install python3.10
curl https://bootstrap.pypa.io/get-pip.py | python3.10
pip3.10 install --upgrade setuptools wheel
pip3.10 install -r requirements.txt
