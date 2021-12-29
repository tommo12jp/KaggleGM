#!/bin/sh

python3 /home/jovyan/work/project/src/train.py --fold 0
python3 /home/jovyan/work/project/src/train.py --fold 1
python3 /home/jovyan/work/project/src/train.py --fold 2
python3 /home/jovyan/work/project/src/train.py --fold 3
python3 /home/jovyan/work/project/src/train.py --fold 4