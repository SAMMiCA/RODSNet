conda activate rodsnet
# Multiply disp to semantic_obstacle channel: use [--disp_to_obst_ch] option
# Multiply (disp+1) to semantic_obstacle channel : use [--disp_plus_1_to_obst_ch] option
# For testing disparity*gamma effects (change gamma value 1 --> 5 or 10) : use [--gamma 1 or 5 or 10] option
# To more focusing on further semantic's loss : use [--with_depth_level_loss] option
# if you want to see semantic and dispairty results images : use [--save_val_results] option (it takes long times to save images..)
# you can set loaded_checkpoint's root location : use [--resume_dir /root/dataset/] option, default : ./
# you can set result's root location : use [--save_dir /root/dataset/] option, default : ./
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_citylost_eps_1e-1_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --with_refine --refinement_type ours \
--batch_size 4 --val_batch_size 4 --train_semantic --train_disparity \
--resume run/city_lost/origin/score_best_checkpoint.pth --epsilon 1e-1 \
--test_only --val_img_height 1024 --val_img_width 2048 --without_depth_range_miou

python main.py --gpu_id 0 --dataset city_lost --model mobilenetv2 --checkname mobilenetv2_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --with_refine --refinement_type ours \
--batch_size 4 --val_batch_size 4 --train_semantic --train_disparity \
--resume run/city_lost/mobilenetv2_train_citylost/experiment_0/score_best_checkpoint.pth --epsilon 1e-1 \
--test_only --val_img_height 1024 --val_img_width 2048 --without_depth_range_miou

python main.py --gpu_id 0 --dataset city_lost --model efficientnetb0 --checkname efficientnetb0_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --with_refine --refinement_type ours \
--batch_size 4 --val_batch_size 4 --train_semantic --train_disparity \
--resume run/city_lost/efficientnetb0_train_citylost/experiment_0/score_best_checkpoint.pth --epsilon 1e-1 \
--test_only --val_img_height 1024 --val_img_width 2048 --without_depth_range_miou

python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new18_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --with_refine --refinement_type new18 \
--batch_size 4 --val_batch_size 4 --train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new18/experiment_1/score_best_checkpoint.pth --epsilon 1e-1 \
--test_only --val_img_height 1024 --val_img_width 2048 --without_depth_range_miou

python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new33_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 --with_refine --refinement_type new33 \
--batch_size 4 --val_batch_size 4 --train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new33/experiment_1/epoch52_checkpoint.pth --epsilon 1e-1 \
--test_only --val_img_height 1024 --val_img_width 2048 --without_depth_range_miou


## Experiment#2 (new1) ---------------------------------------------------------------------------------------------------------
# (change semantic structure in refinement's module to same with disparity's)
# train
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new1_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new1 \
--batch_size 3 --val_batch_size 1 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new1/experiment_2/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only


## Experiment#9 (new4) ---------------------------------------------------------------------------------------------------------
# concat → add
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new4_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new4 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new4/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only


## Experiment#10 (new7)---------------------------------------------------------------------------------------------------------
# disparity map (single channel) * obstacle channel (#20) in semantic class activations
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new7_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new7 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new7_2/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1 --test_only


## Experiment#11 (new8) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of hourglass + (concat → add)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new8_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new8 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new8_con/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only


## experiment#12 (new10) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of stacked hourglass + (concat → add)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new10_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new10 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new10_con/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only



## experiment#19 (new16) ---------------------------------------------------------------------------------------------------------
# #12 + #10  #(add_operation+stacked_hourglass + disparity multiplied)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new16_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new16 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new16/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1 --test_only



## experiment#20 (new17) ---------------------------------------------------------------------------------------------------------
# #10 - tune weight of disparity values by (parameter 5)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new17_param5_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new17 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new17_param5/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 5 --test_only



## experiment#21 (new17) ---------------------------------------------------------------------------------------------------------
# #10 - tune weight of disparity values by (parameter 10)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new17_param10_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new17 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new17_param10/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 10 --test_only


## experiment#22 (new18) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of stacked hourglass  (#concat operation)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new18_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new18 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new18/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only


## experiment#23 (new19) ---------------------------------------------------------------------------------------------------------
# #22 + #10(disp_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new19_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new19 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new19/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1 --test_only


## experiment 25 (new21) ---------------------------------------------------------------------------------------------------------
# #11'(residual connection at start/end of hourglass) + concat + #10(disp_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new21_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new21 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new21/experiment_3/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1 --test_only


## experiment 27 (new23) ---------------------------------------------------------------------------------------------------------
# #10' ( multiply (disparity + 1) to semantic's obstacle channel) : disp_plus_1_to_obst_ch
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new23_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new23 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new23/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1 --test_only


## experiment 28 (new24) ---------------------------------------------------------------------------------------------------------
# #22(res. conn. stacked_hour. concat) + #27(#10_disp+1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new24_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new24 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new24/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1 --test_only



## experiment 31 (new31) ---------------------------------------------------------------------------------------------------------
# #22(stacked concat) + 21'(#10x10 * (disp+1))
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new31_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new31 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new31/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 10 --test_only

## experiment 32 (new32) ---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new32_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new32 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new32/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only \
--resume_dir /root/dataset/ --save_dir /root/dataset/
!![check checkpoint, every epochs checkpoints are saved ]

## experiment 33 (new33)---------------------------------------------------------------------------------------------------------
# 3x stacked hourglass (residual only from input to output)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new33_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new33 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new33/experiment_1/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only \
--resume_dir /root/dataset/ --save_dir /root/dataset/
!![check checkpoint every epochs checkpoints are saved ]


## experiment 34 (new34)---------------------------------------------------------------------------------------------------------
# # original refine's minor change (refine's input disparity + 1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new34_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new34 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new34/experiment_2/score_best_checkpoint.pth \
--epsilon 1e-1 --test_only \
--resume_dir /root/dataset/ --save_dir /root/dataset/
!![check checkpoint every epochs checkpoints are saved, location: /root/dataset/run/city_lost/resnet18_train_refine_new34/experiment_2 ]


## experiment 35 (new35)---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1) + new7__(disp_plus_1_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new35_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new35 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new35/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1 --test_only \
--resume_dir /root/dataset/ --save_dir /root/dataset/
!![check checkpoint every epochs checkpoints are saved, location: /root/dataset/run/city_lost/resnet18_train_refine_new35/experiment_0 ]

## experiment 35__ ---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1) + new7__(disp_plus_1_to_obst_ch)*weight10
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new35_gamma10_test \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new35 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/city_lost/resnet18_train_refine_new35_gamma10/experiment_0/score_best_checkpoint.pth \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 10 --test_only \
--resume_dir /root/dataset/ --save_dir /root/dataset/
!![check checkpoint every epochs checkpoints are saved, location: /root/dataset/run/city_lost/resnet18_train_refine_new35_gamma10/experiment_0 ]
