#!/bin/bash

export model=GASA
export ck=best

echo ${model}

echo 'Middleburry'
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}  --name ${model} --model GASA  --dataset Middlebury --scale 4 --data_root ./data/depth_enhance/01_Middlebury_Dataset 
echo '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 8 --data_root ./data/depth_enhance/01_Middlebury_Dataset
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 16 --data_root ./data/depth_enhance/01_Middlebury_Dataset

echo 'RGBD'
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 4 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 8 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 16 --data_root ./data/depth_enhance/03_RGBD_Dataset

# echo 'RGBZ'
# echo '4'
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Luz --scale 4 --data_root ./data/depth_enhance/02_RGBZ_Dataset
# echo '8'
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}  --name ${model} --model GASA  --dataset Luz --scale 8 --data_root ./data/depth_enhance/02_RGBZ_Dataset
# echo '16'
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Luz --scale 16 --data_root ./data/depth_enhance/02_RGBZ_Dataset

echo 'NYU'
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 4 
echo '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 8
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 16



