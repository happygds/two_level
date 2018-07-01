import torch
from torch import nn

from transforms import *
import torchvision.models

class BinaryClassifier(torch.nn.Module):
    def __init__(self, num_class, course_segment, modality,
                 new_length=None, dropout=0.8, test_mode=False):

        super(BinaryClassifier, self).__init__()
        self.modality = modality
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
        

        self.binary_classifier = nn.Sequential(
            nn.Linear(feature_dim, num_class), nn.Softmax(dim=2))

    def _prepare_binary_classifier(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        self.classifier_fc = nn.Linear(feature_dim, num_class)

        nn.init.normal(self.classifier_fc.weight.data, 0, 0.001)
        nn.init.constant(self.classifier_fc.bias.data, 0)

        self.test_fc = None
        self.feature_dim = feature_dim

        return feature_dim


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
