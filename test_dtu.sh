#!/usr/bin/env bash
CKPT_FILE="./checkpoints/atlas_blended.ckpt"

MVS_DATA="/media/rafweilharter/hard_disk/data/datasets/dtu"
LIST_TEST="/media/rafweilharter/hard_disk/data/datasets/dtu/test.txt"
OUT_DIR="/media/rafweilharter/hard_disk/data/datasets/dtu/outputs"

python test.py --dataset=dtu_full_res --ndepths="32,8,8,8,4" --outdir=$OUT_DIR --ent_high=1.2 --interval_scale=1.0 --input_scale=1.0 --output_scale=2 --consistent=3 --dist=0.25 --rel_dist=100 --neighbors=5 --batch_size=1 --numdepth=384 --testpath=$MVS_DATA --testlist=$LIST_TEST --loadckpt $CKPT_FILE $@
