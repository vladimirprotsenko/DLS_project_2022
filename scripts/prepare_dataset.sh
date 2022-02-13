#!/bin/bash

dataset_path="datasets/small_dataset.zip"

rm -rf datasets/current_dataset
mkdir datasets/current_dataset
unzip -q $dataset_path -d datasets/current_dataset


