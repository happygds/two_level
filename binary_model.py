import torch
from torch import nn

from transforms import *
import torchvision.models

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, input_dim, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()
        self.num_segments = course_segment
        self.course_segment = course_segment
        self.reshape = True
        self.dropout = dropout
        self.test_mode = test_mode
        self.binary_classifier = nn.Linear(input_dim, num_class)

    def forward(self, inputdata, target):
        if not self.test_mode:
            return self.train_forward(inputdata, target)
        else:
            return self.test_forward(inputdata)


    def train_forward(self, inputdata, target):
        course_ft = input_data[:, :, :].mean(dim=1)
        raw_course_ft = self.classifier_fc(course_ft)
        target = target.view(-1)
        return raw_course_ft, target
                

    def test_forward(self, input):
        return self.self.classifier_fc(input), input
