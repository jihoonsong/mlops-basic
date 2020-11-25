# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


class DNN:
    def __init__(self):
        self.estimator = tf.estimator.DNNClassifier(
            feature_columns=[tf.feature_column.numeric_column('image', shape=[28, 28, 1])],
            hidden_units=[1024, 512, 512, 256],
            n_classes=10,
            batch_norm=True
        )

    def get_estimator(self):
        return self.estimator
