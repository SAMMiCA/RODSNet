import os
from glob import glob


def gen_cityscapes():
    data_dir = '/root/dataset/lost_and_found'

    train_file = 'lost_train.txt'
    val_file = 'lost_val.txt'
    dir_name = 'leftImg8bit'

    write_file(train_file, data_dir, dir_name, mode='train')
    write_file(val_file, data_dir, dir_name, mode='val')


def write_file(file, data_dir, dir_name, mode='train'):
    with open(file, 'w') as f:
        left_dir = os.path.join(data_dir, dir_name, mode)
        left_imgs = recursive_glob(left_dir, suffix='.png')
        left_imgs.sort()
        print('Number of {} images: {}'.format(mode, len(left_imgs)))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'rightImg8bit')
            disp_path = left_img.replace(dir_name, 'disparity')
            label_path = left_img.replace(dir_name, 'gtCoarse')
            label_path = label_path.replace('.png', '_labelTrainIds.png')

            f.write(left_img.replace(data_dir + '/', '') + ' ')
            f.write(right_img.replace(data_dir + '/', '') + ' ')
            f.write(disp_path.replace(data_dir + '/', '') + ' ')
            f.write(label_path.replace(data_dir + '/', '') + '\n')


def recursive_glob( rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

if __name__ == '__main__':
    gen_cityscapes()

