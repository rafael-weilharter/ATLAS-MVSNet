#!/usr/bin/env bash
MVS_TRAINING="/media/rafweilharter/hard_disk/data/datasets/dtu"
LIST_TRAIN="/media/rafweilharter/hard_disk/data/datasets/dtu/train.txt"
LIST_TEST="/media/rafweilharter/hard_disk/data/datasets/dtu/test.txt"

python train.py --dataset=dtu_full_res --batch_size=1 --input_scale=1.0 --output_scale=2 --ndepths="32,8,8,8,4" --interval_scale=1.0 --trainpath=$MVS_TRAINING --trainlist=$LIST_TRAIN --testlist=$LIST_TEST --numdepth=384 --logdir ./checkpoints/new_training $@
