#!/usr/bin/env bash
datapath="/home/devil/Downloads/UniMVSNet-main/datasets/lists/dtu/"
outdir="/home/devil/Downloads/UniMVSNet-main/out_put/"
resume="/home/devil/Downloads/UniMVSNet-main/model/unimvsnet_dtu.ckpt"
fusibile_exe_path="/home/devil/Downloads/fusibile-master/fusibile"

CUDA_VISIBLE_DEVICES=0 python main.py \
        --test \
        --ndepths 48 32 8 \
        --interval_ratio 4 2 1 \
        --max_h 864 \
        --max_w 1152 \
        --num_view 5 \
        --outdir $outdir \
        --datapath $datapath \
        --resume $resume \
        --dataset_name "general_eval" \
        --batch_size 1 \
        --testlist "datasets/lists/dtu/test.txt" \
        --fea_mode "fpn" \
        --agg_mode "adaptive" \
        --depth_mode "unification" \
        --numdepth 192 \
        --interval_scale 1.06 \
        --filter_method "gipuma" \
        --fusibile_exe_path $fusibile_exe_path \
        --prob_threshold 0.3 \
        --disp_threshold 0.25 \
        --num_consistent 3 ${@:1}
