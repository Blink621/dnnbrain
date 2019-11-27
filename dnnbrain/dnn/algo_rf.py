# import some necessary packages
import numpy as np
from os import remove
import torch, copy, cv2
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from dnnbrain.dnn.core import Mask, Algorithm
from dnnbrain.dnn import models as db_models # Use eval to import model model



class OccluderDiscrepancyMapping(Algorithm):

    """
    An class to compute activation for each pixel from an image
    using slide Occluder
    """

    def __init__(self, model, layer, channel, window=(11, 11), stride=(2, 2), metric='max'):
        super(UpsamplingActivationMapping, self).__init__(model, layer, channel)
        self.window = window
        self.stride = stride
        self.metric = metric

    def set_params(self, window=(11, 11), stride=(2, 2), metric='max'):

        """
        Set necessary parameters for the estimator

        Parameter:
        ---------
        window[list]: The size of sliding window, which form should be [int, int].
        The window will start from top-left and slides from left to right,
        and then from top to bottom.

        stride[list]: The move step if sliding window, which form should be [int, int]
        The first element of stride is the step for rows, while the second element of
        stride is the step for column.

        metric[str]: The metric to measure how feature map change, max or mean.
        """

        self.window = window
        self.stride = stride
        self.metric = metric

    def compute(self, image):

        """
        Please implement the sliding occluder algothrim for discrepancy map.
        """

        self.croped_img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        self.croped_img = self.croped_img.transpose(2, 0, 1)[np.newaxis, :]

        widenum = int((self.croped_img.shape[2] - self.window[0]) / self.stride[0] + 1)
        heightnum = int((self.croped_img.shape[3] - self.window[1]) / self.stride[0] + 1)

        discrepancy_map = np.zeros((widenum, heightnum))
        discrepancy_map_whole = np.max(self.model.compute_activation(self.croped_img, self.mask).get(self.channel))

        r = 1

        for i in range(0, widenum):
            for j in range(0, heightnum):
                tmpoc = copy.deepcopy(self.croped_img)
                tmpoc[self.stride[0] * i:self.stride[0] * i + self.window[0],
                      self.stride[1] * j:self.stride[1] * j + self.window[1], :] = 0
                tmpoc = tmpoc.transpose(2, 0, 1)[np.newaxis, :]
                tmpmax = np.max(self.model.compute_activation(tmpoc, self.mask).get(self.channel))
                discrepancy_map[i, j] = discrepancy_map_whole - tmpmax

                print(r, 'in', widenum * heightnum,
                      'finished. Discrepancy: %.1f' % abs(discrepancy_map[i, j]))
                r = r + 1

        return discrepancy_map


class UpsamplingActivationMapping(Algorithm):

    """
        A class to compute activation for each pixel from an image by upsampling
        activation map, with specific method assigned.
    """

    def __init__(self, model, layer, channel, interp_meth='bicubic', interp_threshold=0.68):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        interp_threshold[int]: The threshold to filter the feature map,
                               which you should assign between 0-99.
        """

        super(UpsamplingActivationMapping, self).__init__(model, layer, channel)
        self.mask = Mask()
        self.mask.set(self.layer, [self.channel, ])
        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def set_params(self, interp_meth='bicubic', interp_threshold=0.68):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        interp_threshold[int]: The threshold to filter the feature map,
                    which you should assign between 0-99.
        """

        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def compute(self, image):

        """
        The method do real computation for pixel activation based on feature mapping upsampling.

        Parameter:
        ---------
        image[np-array]: a W x H x 3 Numpy Array.
        """

        croped_img = cv2.resize(image, self.model.img_size, interpolation=cv2.INTER_CUBIC)
        self.croped_img = croped_img.transpose(2, 0, 1)[np.newaxis, :]
        self.img_act = self.model.compute_activation(self.croped_img, self.mask).get(self.layer).squeeze()
        self.img_act = torch.from_numpy(self.img_act)[np.newaxis, np.newaxis, ...]
        self.img_act = interpolate(self.img_act, size=self.croped_img.shape[2:4],
                                   mode=self.interp_meth, align_corners=True)
        self.img_act = np.squeeze(np.asarray(self.img_act))
        thresed_img_act = copy.deepcopy(self.img_act)

        thresed_img_act[thresed_img_act < np.percentile(thresed_img_act, self.interp_threshold * 100)] = 0
        thresed_img_act = thresed_img_act / np.max(thresed_img_act)
        self.thresed_img_act = thresed_img_act
        return self.thresed_img_act


class EmpiricalReceptiveField():

    """
    A class to estimate empiral receptive field of a DNN model.
    """

    def __init__(self, model=None, layer=None, channel=None, threshold=0.3921):

        """
        Parameter:
        ---------
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.
        """

        super(EmpiricalReceptiveField, self).__init__(model, layer, channel)
        self.model = model
        self.threshold = threshold

    def set_params(self, threshold=0.3921):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.
        """

        self.threshold = threshold

    def generate_rf(self, all_thresed_act):

        """
        Compute RF on provided image for target layer and channel.

        Parameter:
        ---------
        all_thresed_act[n x w x h]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.

        """

        self.all_thresed_act = all_thresed_act
        sum_act = np.zeros([self.all_thresed_act.shape[0], self.model.img_size[0] * 2 - 1,
                            self.model.img_size[1] * 2 - 1])
        for pics_layer in range(self.all_thresed_act.shape[0]):
            cx = int(np.mean(np.where(self.all_thresed_act[pics_layer, :, :] ==
                                      np.max(self.all_thresed_act[pics_layer, :, :]))[0]))
            cy = int(np.mean(np.where(self.all_thresed_act[pics_layer, :, :] ==
                                      np.max(self.all_thresed_act[pics_layer, :, :]))[1]))
            sum_act[pics_layer, self.model.img_size[0] - 1 - cx:2 * self.model.img_size[0] - 1 - cx,
                    self.model.img_size[1] - 1 - cy:2 * self.model.img_size[1] - 1 - cy] = \
                self.all_thresed_act[pics_layer, :, :]

        sum_act = np.sum(sum_act, 0)[int(self.model.img_size[0] / 2):int(self.model.img_size[0] * 3 / 2),
                                     int(self.model.img_size[1] / 2):int(self.model.img_size[1] * 3 / 2)]
        plt.imsave('tmp.png', sum_act, cmap='gray')
        rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
        remove('tmp.png')
        rf = cv2.medianBlur(rf, 31)
        _, th = cv2.threshold(rf, self.threshold * 255, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # rf = cv2.ellipse(rf, cv2.fitEllipse(contours[0]), (255, 255, 255), 1, 300)
        rf_contour = np.array(contours).squeeze()
        empirical_rf_area = 0
        for i in np.unique(rf_contour[:, 0]):
            empirical_rf_area = empirical_rf_area + max(rf_contour[rf_contour[:, 0] == i, 1]) - \
                min(rf_contour[rf_contour[:, 0] == i, 1])
        empirical_rf_size = np.sqrt(empirical_rf_area)
        return empirical_rf_size


class TheoreticalReceptiveField(Algorithm):
    """
    A class to count theoretical receptive field. Noted that now only AlexNet,
    Vgg16, Vgg19 are supported (all these net are linear structure).
    """
    def compute(self):
        if self.model.__class__.__name__ == 'AlexNet':
            self.net_struct = {}
            self.net_struct['net'] = [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3',
                                       'conv4', 'conv5', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg11':
            self.net_struct = {}
            self.net_struct['net'] = [[3, 1, 1], [2, 2, 0], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2',
                                       'conv3_1', 'conv3_2', 'pool3', 'conv4_1',
                                       'conv4_2', 'pool4', 'conv5_1', 'conv5_2',
                                       'pool5']

        if self.dnn.__class__.__name__ == 'Vgg16':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2',
                                       'conv4_3', 'pool4', 'conv5_1', 'conv5_2',
                                       'conv5_3', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg19':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'conv3_4', 'pool3', 'conv4_1',
                                       'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
                                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                                       'pool5']
        theoretical_rf_size = 1
        for layer in reversed(range(self.net_struct['name'].index(self.layer) + 1)):
            fsize, stride, padding = self.net_struct['net'][layer]
            theoretical_rf_size = ((theoretical_rf_size - 1) * stride) + fsize
        return theoretical_rf_size
