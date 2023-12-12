 #!/bin/bash

###parameters for the scenes

##NeRF-Synthetic
SCALE=(0.8 0.8 0.7 0.6 0.8 0.7 0.8 0.8)
BOUND=(1.0 1.0 1.0 0.7 1.0 1.0 1.0 1.0)
TEST=("lego" "drums" "hotdog" "materials" "mic" "ship" "chair" "ficus")
ITERS=(15000 10000)
FEW_SHOT=(0 8)
DIFF_REG_START_ITER=(2000 1000)


### scenes
### 8 scenes (0, 1, ..., 7)
### Choose the exact index to run only a specific scene (see the previous "TEST" variable)
for i_dataset in 0 1 2 3 4 5 6 7
do

### folders for dataset and output base on the chosen scenes and few-shot setting

##synthetic
DATASET="data/nerf_synthetic/"${TEST[i_dataset]}
WORK_BASE=("test_synthetic/test_"${TEST[i_dataset]} "test_synthetic/test_"${TEST[i_dataset]}"/few_shot8")
DATASET_NAME=(${TEST[i_dataset]} ${TEST[i_dataset]}" 8-views")

case ${TEST[i_dataset]} in

  hotdog)
    AABB_BOX=(-1.0 -0.6 -1.0 1.0 1.0 1.0)
    ;;

  ship)
    AABB_BOX=(-1.0 -0.5 -1.0 1.0 0.5 1.0)
    ;;

  *)
    AABB_BOX=(-${BOUND[i_dataset]} -${BOUND[i_dataset]} -${BOUND[i_dataset]} ${BOUND[i_dataset]} ${BOUND[i_dataset]} ${BOUND[i_dataset]})
    ;;
esac


### few-shot setting
### NeRF-Synthetic:  0--> no few-shot; 1--> 8-views
### Choose a specific index to run only a specific few-shot setting
for i_shot in 1
do


### Run NeRF-Synthetic

# training
CUDA_VISIBLE_DEVICES=0 python main_nerf.py $DATASET --workspace ${WORK_BASE[i_shot]}/test_CombiNeRF --fp16 --few_shot ${FEW_SHOT[i_shot]} --scale ${SCALE[i_dataset]} --bound ${BOUND[i_dataset]} --aabb_box ${AABB_BOX[0]} ${AABB_BOX[1]} ${AABB_BOX[2]} ${AABB_BOX[3]} ${AABB_BOX[4]} ${AABB_BOX[5]} --dataset_name "${DATASET_NAME[i_shot]}" --implementation_name "CombiNeRF" --iters ${ITERS[i_shot]} --dt_gamma 0 --num_rays 7008 --max_ray_batch 7008 --downscale 2 --num_levels 32 --diff_reg --loss_dist --loss_fg --use_depth --diff_reg_start_iter ${DIFF_REG_START_ITER[i_shot]} --dist_lambda 2e-3 --fg_lambda 1e-3 --patch_size 4 --depth_reg_lambda 0.01 --use_lipshitz_color --use_lipshitz_sigma --freq_reg_mask_sigma --freq_reg_mask_color --total_iter_end_rate 0.2 --smoothing --smoothing_lambda 0.00001 --smooth_sampling_method near_pixel --num_testval_images 25

# test (taking the best checkpoint during training)
CUDA_VISIBLE_DEVICES=0 python main_nerf.py $DATASET --workspace ${WORK_BASE[i_shot]}/test_CombiNeRF --fp16 --few_shot ${FEW_SHOT[i_shot]} --scale ${SCALE[i_dataset]} --bound ${BOUND[i_dataset]} --aabb_box ${AABB_BOX[0]} ${AABB_BOX[1]} ${AABB_BOX[2]} ${AABB_BOX[3]} ${AABB_BOX[4]} ${AABB_BOX[5]} --dataset_name "${DATASET_NAME[i_shot]}" --implementation_name "CombiNeRF" --iters ${ITERS[i_shot]} --dt_gamma 0 --num_rays 7008 --max_ray_batch 7008 --downscale 2 --num_levels 32 --diff_reg --loss_dist --loss_fg --use_depth --diff_reg_start_iter ${DIFF_REG_START_ITER[i_shot]} --dist_lambda 2e-3 --fg_lambda 1e-3 --patch_size 4 --depth_reg_lambda 0.01 --use_lipshitz_color --use_lipshitz_sigma --freq_reg_mask_sigma --freq_reg_mask_color --total_iter_end_rate 0.2 --smoothing --smoothing_lambda 0.00001 --smooth_sampling_method near_pixel --test --num_testval_images 25 --ckpt best


done

done

