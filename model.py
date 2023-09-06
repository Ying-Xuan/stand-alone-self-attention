import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50, resnet101, resnet152
from attention import Attention


def get_model(base_model=resnet50(), num_classses=1000):

    for i in range(len(base_model.layer1)):
        base_model.layer1[i].conv2 = Attention(in_channel=base_model.layer1[i].conv2.in_channels, out_channel=base_model.layer1[i].conv2.out_channels, 
                                        kernel_size=3, stride=base_model.layer1[i].conv2.stride)
    
    for i in range(len(base_model.layer2)):
        base_model.layer2[i].conv2 = Attention(in_channel=base_model.layer2[i].conv2.in_channels, out_channel=base_model.layer2[i].conv2.out_channels, 
                                        kernel_size=3, stride=base_model.layer2[i].conv2.stride)
    
    for i in range(len(base_model.layer3)):
        base_model.layer3[i].conv2 = Attention(in_channel=base_model.layer3[i].conv2.in_channels, out_channel=base_model.layer3[i].conv2.out_channels, 
                                        kernel_size=3, stride=base_model.layer3[i].conv2.stride)
    
    for i in range(len(base_model.layer4)):
        base_model.layer4[i].conv2 = Attention(in_channel=base_model.layer4[i].conv2.in_channels, out_channel=base_model.layer4[i].conv2.out_channels, 
                                        kernel_size=3, stride=base_model.layer4[i].conv2.stride)
        
    fc_features = base_model.fc.in_features 
    base_model.fc = nn.Linear(fc_features, num_classses)

    return base_model


def main():

    net = get_model(resnet50(), num_classses=2)
    print(net)

if __name__ == '__main__':
    main()