conda activate rodsnet

# multiply disp to semantic_obstacle channel: use [--disp_to_obst_ch] option
# multiply (disp+1) to semantic_obstacle channel : use [--disp_plus_1_to_obst_ch] option
# For testing disparity*gamma effects (change gamma value 1 --> 5 or 10) : use [--gamma 1 or 5 or 10] option
# to more focusing on further semantic's loss : use [--with_depth_level_loss] option
# If you want to save every epoch's checkpoint : use [--save_pth_every_epoch] option


## Experiment#2 (new1) ---------------------------------------------------------------------------------------------------------
# (change semantic structure in refinement's module to same with disparity's)
# train
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new1 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new1 \
--batch_size 3 --val_batch_size 1 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1



## Experiment#9 (new4) ---------------------------------------------------------------------------------------------------------
# concat → add
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new4 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new4 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1


## Experiment#10 (new7)---------------------------------------------------------------------------------------------------------
# disparity map (single channel) * obstacle channel (#20) in semantic class activations
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new7 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new7 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1


## Experiment#11 (new8) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of hourglass + (concat → add)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new8 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new8 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1


## experiment#12 (new10) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of stacked hourglass + (concat → add)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new10 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new10 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1



## experiment#19 (new16) ---------------------------------------------------------------------------------------------------------
# #12 + #10  #(add_operation+stacked_hourglass + disparity multiplied)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new16 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new16 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1



## experiment#20 (new17) ---------------------------------------------------------------------------------------------------------
# #10 - tune weight of disparity values by (parameter 5)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new17_param5 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new17 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 5



## experiment#21 (new17) ---------------------------------------------------------------------------------------------------------
# #10 - tune weight of disparity values by (parameter 10)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new17_param10 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new17 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 10


## experiment#22 (new18) ---------------------------------------------------------------------------------------------------------
# residual connection at start/end of stacked hourglass  (#concat operation)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new18 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new18 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1


## experiment#23 (new19) ---------------------------------------------------------------------------------------------------------
# #22 + #10(disp_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new19 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new19 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1


## experiment 25 (new21) ---------------------------------------------------------------------------------------------------------
# #11'(residual connection at start/end of hourglass) + concat + #10(disp_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new21 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new21 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_to_obst_ch --gamma 1


## experiment 27 (new23) ---------------------------------------------------------------------------------------------------------
# #10' ( multiply (disparity + 1) to semantic's obstacle channel) : disp_plus_1_to_obst_ch
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new23 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new23 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1


## experiment 28 (new24) ---------------------------------------------------------------------------------------------------------
# #22(res. conn. stacked_hour. concat) + #27(#10_disp+1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new24 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new24 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1



## experiment 31 (new31) ---------------------------------------------------------------------------------------------------------
# #22(stacked concat) + 21'(#10x10 * (disp+1))
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new31 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new31 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 10

## experiment 32 (new32) ---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new32 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new32 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--save_pth_every_epoch --save_dir /root/dataset/


## experiment 33 (new33)---------------------------------------------------------------------------------------------------------
# 3x stacked hourglass (residual only from input to output)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new33 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new33 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--save_pth_every_epoch --save_dir /root/dataset/


## experiment 34 (new34)---------------------------------------------------------------------------------------------------------
# # original refine's minor change (refine's input disparity + 1)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new34 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new34 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--save_pth_every_epoch --save_dir /root/dataset/


## experiment 35 (new35)---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1) + new7__(disp_plus_1_to_obst_ch)
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new35 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new35 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 1 \
--save_pth_every_epoch --save_dir /root/dataset/


## experiment 35__ ---------------------------------------------------------------------------------------------------------
# #22'' stacked hourglass (concat) & (refine's input disparity + 1) + new7__(disp_plus_1_to_obst_ch)*weight10
python main.py --gpu_id 0 --dataset city_lost --model resnet18 --checkname resnet18_train_refine_new35_gamma10 \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--with_refine --refinement_type new35 \
--batch_size 4 --val_batch_size 4 \
--train_semantic --train_disparity \
--resume run/sceneflow/best_disp_model/epe_best_checkpoint.pth --transfer_disparity \
--epsilon 1e-1 \
--disp_plus_1_to_obst_ch --gamma 10 \
--save_pth_every_epoch --save_dir /root/dataset/
