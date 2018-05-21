#!/bin/bash

PATH_TO_CONFIG="/home/f/PycharmProjects/AplicIndu/P3/ssd_mobilenet_v1_pet.config"
PATH_TO_TRAINING="/home/f/PycharmProjects/AplicIndu/P3/training/tensorboard"

export PYTHONPATH=$PYTHONPATH:/home/f/PycharmProjects/code/models/research:/home/f/PycharmProjects/code/models/research/slim

python /home/f/PycharmProjects/code/models/research/object_detection/train.py --logtostderr --train_dir $PATH_TO_TRAINING --pipeline_config_path $PATH_TO_CONFIG
