import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


class TimeAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Evaluator(object):
    def __init__(self, num_class, opts):
        self.num_class = num_class
        self.opts = opts
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)
        if not self.opts.without_depth_range_miou:
            self.confusion_matrix_depth = {'20': np.zeros((self.num_class,) * 2),
                                           '40': np.zeros((self.num_class,) * 2),
                                           '60': np.zeros((self.num_class,) * 2),
                                           '80': np.zeros((self.num_class,) * 2),
                                           '100': np.zeros((self.num_class,) * 2)}

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each classes-----------')
        print("road         : %.6f" % (Acc[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (Acc[1] * 100.0), "%\t")
        print("building     : %.6f" % (Acc[2] * 100.0), "%\t")
        print("wall         : %.6f" % (Acc[3] * 100.0), "%\t")
        print("fence        : %.6f" % (Acc[4] * 100.0), "%\t")
        print("pole         : %.6f" % (Acc[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (Acc[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (Acc[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (Acc[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (Acc[9] * 100.0), "%\t")
        print("sky          : %.6f" % (Acc[10] * 100.0), "%\t")
        print("person       : %.6f" % (Acc[11] * 100.0), "%\t")
        print("rider        : %.6f" % (Acc[12] * 100.0), "%\t")
        print("car          : %.6f" % (Acc[13] * 100.0), "%\t")
        print("truck        : %.6f" % (Acc[14] * 100.0), "%\t")
        print("bus          : %.6f" % (Acc[15] * 100.0), "%\t")
        print("train        : %.6f" % (Acc[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (Acc[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (Acc[18] * 100.0), "%\t")
        if self.num_class == 20:
            print("small obstacles: %.6f" % (Acc[19] * 100.0), "%\t")
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self, save_filename):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        print("road         : %.6f" % (MIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (MIoU[1] * 100.0), "%\t")
        print("building     : %.6f" % (MIoU[2] * 100.0), "%\t")
        print("wall         : %.6f" % (MIoU[3] * 100.0), "%\t")
        print("fence        : %.6f" % (MIoU[4] * 100.0), "%\t")
        print("pole         : %.6f" % (MIoU[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (MIoU[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (MIoU[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (MIoU[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (MIoU[9] * 100.0), "%\t")
        print("sky          : %.6f" % (MIoU[10] * 100.0), "%\t")
        print("person       : %.6f" % (MIoU[11] * 100.0), "%\t")
        print("rider        : %.6f" % (MIoU[12] * 100.0), "%\t")
        print("car          : %.6f" % (MIoU[13] * 100.0), "%\t")
        print("truck        : %.6f" % (MIoU[14] * 100.0), "%\t")
        print("bus          : %.6f" % (MIoU[15] * 100.0), "%\t")
        print("train        : %.6f" % (MIoU[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (MIoU[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (MIoU[18] * 100.0), "%\t")
        if self.num_class == 20:
            print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

        # Save validation results
        with open(save_filename, 'a') as f:
            f.write('-----------IoU of each classes-----------')
            f.write("road         : %.6f \n" % (MIoU[0] * 100.0))
            f.write("sidewalk     : %.6f \n" % (MIoU[1] * 100.0))
            f.write("building     : %.6f\n" % (MIoU[2] * 100.0))
            f.write("wall         : %.6f\n" % (MIoU[3] * 100.0))
            f.write("fence        : %.6f\n" % (MIoU[4] * 100.0))
            f.write("pole         : %.6f\n" % (MIoU[5] * 100.0))
            f.write("traffic light: %.6f\n" % (MIoU[6] * 100.0))
            f.write("traffic sign : %.6f\n" % (MIoU[7] * 100.0))
            f.write("vegetation   : %.6f\n" % (MIoU[8] * 100.0))
            f.write("terrain      : %.6f\n" % (MIoU[9] * 100.0))
            f.write("sky          : %.6f\n" % (MIoU[10] * 100.0))
            f.write("person       : %.6f\n" % (MIoU[11] * 100.0))
            f.write("rider        : %.6f\n" % (MIoU[12] * 100.0))
            f.write("car          : %.6f\n" % (MIoU[13] * 100.0))
            f.write("truck        : %.6f\n" % (MIoU[14] * 100.0))
            f.write("bus          : %.6f\n" % (MIoU[15] * 100.0))
            f.write("train        : %.6f\n" % (MIoU[16] * 100.0))
            f.write("motorcycle   : %.6f\n" % (MIoU[17] * 100.0))
            f.write("bicycle      : %.6f\n" % (MIoU[18] * 100.0))
            if self.num_class == 20:
                f.write("small obstacles: %.6f\n" % (MIoU[19] * 100.0))

        obs_mIoU = MIoU[19]
        MIoU = np.nanmean(MIoU)
        return MIoU, obs_mIoU

    def Mean_Intersection_over_Union_with_depth(self, d_range, save_filename):
        cf_matrix = self.confusion_matrix_depth[str(d_range)]

        MIoU = np.diag(cf_matrix) / (
                    np.sum(cf_matrix, axis=1) + np.sum(cf_matrix, axis=0) -
                    np.diag(cf_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        if self.num_class == 20:
            print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

            with open(save_filename, 'a') as f:
                f.write("small obstacles: %.6f \n" % (MIoU[19] * 100.0))

        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def _generate_matrix_with_depth(self, gt_image, pre_image, mask):
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch_with_depth(self, gt_image, pre_image, disp):
        assert gt_image.shape == pre_image.shape
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        depth_range = [20, 40, 60, 80, 100]

        self.confusion_matrix += self._generate_matrix_with_depth(gt_image, pre_image, mask)

        for d_range in depth_range:
            if d_range == 20:
                disp_mask = mask & (disp >= 483 / d_range)  # focal_length*base_line : 2300pixel * 0.21m
            else:
                disp_mask = mask & (disp >= 483 / d_range) & (disp < (483 / (d_range - 20)))
            self.confusion_matrix_depth[str(d_range)] += self._generate_matrix_with_depth(gt_image, pre_image, disp_mask)


    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        obs_iu = iu[19]
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_class), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Obstacle IoU": obs_iu,
            }

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        return string

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        if not self.opts.without_depth_range_miou:
            self.confusion_matrix_depth = {'20': np.zeros((self.num_class,) * 2),
                                           '40': np.zeros((self.num_class,) * 2),
                                           '60': np.zeros((self.num_class,) * 2),
                                           '80': np.zeros((self.num_class,) * 2),
                                           '100': np.zeros((self.num_class,) * 2)}
