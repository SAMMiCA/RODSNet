import os
from glob import glob


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def gen_scene_flow_driving_train():
    data_filenames = 'SceneFlow_finalpass_train.txt'
    lines = read_text_lines(data_filenames)

    train_file = 'SceneFlow_finalpass_Driving_train.txt'
    # val_file = 'kitti_2015/KITTI_2015_val.txt'

    with open(train_file, 'w') as train_f:
        for line in lines:
            splits = line.split()

            left_img, right_img, gt_disp = splits[:3]
            Data_type = left_img.split('/')[0]
            if Data_type == 'Driving':
                train_f.write(left_img + ' ')
                train_f.write(right_img + ' ')
                train_f.write(gt_disp + '\n')


if __name__ == '__main__':
    gen_scene_flow_driving_train()

