import os
from glob import glob


def gen_cityscapes():
    data_dir = '/home/tjsong/datasets/cityscapes'

    train_file = 'cityscapes_semantic_train.txt'
    val_file = 'cityscapes_semantic_val.txt'
    test_file = 'cityscapes_semantic_test.txt'
    dir_name = 'leftImg8bit'

    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    write_file(train_file, data_dir, dir_name, mode='train')
    write_file(val_file, data_dir, dir_name, mode='val')
    write_file(test_file, data_dir, dir_name, mode='test')


def write_file(file, data_dir, dir_name, mode='train'):
    with open(file, 'w') as f:
        left_dir = os.path.join(data_dir, dir_name, mode)
        left_imgs = recursive_glob(left_dir, suffix='.png')
        left_imgs.sort()
        print('Number of {} images: {}'.format(mode, len(left_imgs)))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'rightImg8bit')
            disp_path = left_img.replace(dir_name, 'disparity')
            label_path = left_img.replace(dir_name, 'gtFine')
            label_path = label_path.replace('.png', '_labelIds.png')

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
