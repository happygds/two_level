import torch
from torch import nn

from transforms import *
import torchvision.models

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment,
                 new_length=None, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()
        self.num_segments = course_segment
        self.course_segment = course_segment
        self.reshape = True
        self.dropout = dropout
        self.test_mode = test_mode

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        print(("""
               Initializing BinaryClassifier with base model:{}
               BinaryClassifier Configurations:
                   input_modality: {}
                   course_segment: {}
                   num_segments:   {}
                   new_length:     {}
                   dropout_ratio:  {}
                   bn_mode:        {}
              """.format(base_model, self.modality, self.course_segment, self.num_segments,
                         self.new_length, self.dropout)))
        

        self.binary_classifier = nn.Linear(feature_dim, num_class)

    def forward(self, inputdata, target):
        if not self.test_mode:
            return self.train_forward(inputdata, target)
        else:
            return self.test_forward(inputdata)


    def train_forward(self, inputdata, target):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        base_out = self.base_model(inputdata.view((-1, sample_len) + inputdata.size()[-2:]))
        src = base_out.view(-1, self.course_segment, base_out.size()[1])
        course_ft = src[:, :, :].mean(dim=1)
        raw_course_ft = self.classifier_fc(course_ft)
        target = target.view(-1)

        return raw_course_ft, target
                

    def test_forward(self, input):
        sample_len = (3 if self.modality == 'RGB' else 2) * self.new_length
        base_out = self.base_model(input.view((-1,sample_len) + input.size()[-2:]))
        return self.test_fc(base_out), base_out
