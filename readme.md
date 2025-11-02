## Echo-Vision Foundation Model

This repository contains the codebase for pretraining and main training of the model. The pretraining phase focuses on learning general representations, while the main training phase fine-tunes the model for downstream tasks.

## Pretraining 

This script supports pre-training model with DDP by Accelerator library. We use MIMIC-ECHO as pre-training data.

`accelerate launch --config_file ./config_fourgpus.yaml prertain.py [-option]`

## Fine-tuning on downstream tasks
Downstream datasets include echonet_dynamic, camus and tmed.

`accelerate launch --config_file ./config_twogpus.yaml train_main.py [-option]`

## Pre-trained checkpoint

