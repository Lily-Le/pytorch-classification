from __future__ import absolute_import

import pdb

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
from models.auxnet.auxiliary_nets import  AuxClassifier
from models.auxnet.configs import Aux

__all__ = ['resnet_loc']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_aux(nn.Module):

    def __init__(self, depth, ratio,num_classes=10, block_name='BasicBlock',arch='resnet110',layer_loc=5,wide_list=(16, 16, 32, 64), \
                 aux_net_config='1c2f', local_loss_mode='cross_entropy',
                 aux_net_widen=1, aux_net_feature_dim=128
                 ):
        super(ResNet_aux, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.aux_ratio = ratio
        self.aux_layer_loc=layer_loc
        self.criterion_ce = nn.CrossEntropyLoss()


        self.layers = [n,n,n]
        try:
            self.aux_config = Aux[0][arch][layer_loc]
        except:
            raise NotImplementedError

        for item in self.aux_config:
            module_index, layer_index = item

            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                 'loss_mode=local_loss_mode, class_num=num_classes, '
                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_original(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, img, target=None):

        # if self.training:
            stage_i = 0
            layer_i = 0
            # local_module_i = 0

            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            loss_ixy=[]
            aux_features=[]
            # if local_module_i <= self.local_module_num - 2:
            if self.aux_config[0][0] == stage_i \
                    and self.aux_config[0][1] == layer_i:
                # ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                # ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                # ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                # loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
                loss_ixy_,aux_feature_ = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(x, target)
                loss_ixy.append(loss_ixy_)
                aux_features.append(aux_feature_)


                # local_module_i += 1

            for stage_i in (1, 2, 3):  # stage_i :usd to
                for layer_i in range(self.layers[stage_i - 1]):
                    x = eval('self.layer' + str(stage_i))[layer_i](x)

                    # if local_module_i <= self.local_module_num - 2:
                    if self.aux_config[0][0] == stage_i \
                            and self.aux_config[0][1] == layer_i:
                        # ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                        # ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                        # ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                        # loss_ixx: reconstruction loss?
                        # img: 1024*3*32*32
                        # x: 1024*32*16*16
                        # loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(
                        #     img))
                        loss_ixy_, aux_feature_ = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(
                            x, target)
                        loss_ixy.append(loss_ixy_)
                        aux_features.append(aux_feature_)                            # loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                        # loss.backward()
                        # x = x.detach()
                        # local_module_i += 1

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            loss_e2e = self.criterion_ce(logits, target)
            assert len(loss_ixy)==1, f'shape of loss_ixy is {len(loss_ixy)}'

            loss = loss_e2e + self.aux_ratio * loss_ixy[0]
            if self.training:
                loss.backward()
            return logits, aux_features[0],loss_e2e,loss_ixy[0],loss

        # else:
        #     x = self.conv1(img)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        #
        #     x = self.layer1(x)
        #     x = self.layer2(x)
        #     x = self.layer3(x)
        #
        #     x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     logits = self.fc(x)
        #     loss_e2e = self.criterion_ce(logits, target)
        #     return logits, loss,loss_e2e,loss_ixy


def resnet_loc(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet_aux(**kwargs)
