"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import numpy as np

from ..common import NumericalValue, ListValue, StringValue, softmax

from .image_model_detection import ImageModel


class Detection(ImageModel):
    __model__ = 'Detection'

    def __init__(self, network_info, configuration=None):
        super().__init__(network_info, configuration)
        self._check_io_number(1, 4)
        if self.path_to_labels:
            self.labels = self._load_labels(self.path_to_labels)
        self.boxes_name, self.classes_name, self.scores_name, self.detnum_name = self._get_outputs()
        
    def _load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = []
            for s in f:
                begin_idx = s.find(' ')
                if (begin_idx == -1):
                    self.raise_error('The labels file has incorrect format.')
                end_idx = s.find(',')
                labels.append(s[(begin_idx + 1):end_idx])
        return labels

    def _get_outputs(self):
        outputs_set = iter(self.outputs)
        boxes_name = next(outputs_set)
        labels_name = next(outputs_set)
        scores_name = next(outputs_set)
        detnum_name = next(outputs_set)

        #if len(layer_shape) != 2 and len(layer_shape) != 4:
        #    self.raise_error('The Detection model wrapper supports topologies only with 2D or 4D output')
        #if len(layer_shape) == 4 and (layer_shape[2] != 1 or layer_shape[3] != 1):
        #    self.raise_error('The Detection model wrapper supports topologies only with 4D '
        #                     'output which has last two dimensions of size 1')
        #if self.labels:
        #    if (layer_shape[1] == len(self.labels) + 1):
        #        self.labels.insert(0, 'other')
        #        self.logger.warning("\tInserted 'other' label as first.")
        #    if layer_shape[1] != len(self.labels):
        #        self.raise_error("Model's number of classes and parsed "
        #                         'labels must match ({} != {})'.format(layer_shape[1], len(self.labels)))
        return boxes_name, labels_name, scores_name, detnum_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('crop')
        parameters.update({
            'topk': NumericalValue(value_type=int, default_value=1, min=1),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            ),
        })
        return parameters

    def postprocess(self, outputs, meta):
        detnum = int(outputs[self.detnum_name][0])
        bboxes = outputs[self.boxes_name].squeeze()[:detnum]
        labels = outputs[self.classes_name].squeeze()[:detnum]
        scores = outputs[self.scores_name].squeeze()[:detnum]

        return list(zip(bboxes, labels, scores))
