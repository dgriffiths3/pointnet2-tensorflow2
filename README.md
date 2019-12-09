# Pointnet++ tensorflow 2.0 layers

The repository contains implementations of the pointnet++ set abstraction and feature propagation layers as `tf.keras.layers` classes. The intention is not to be a full pointnet++ tensorflow 2.0 implementation, but provide an easy way to build a pointnet++ style network architecture using the tensorflow 2.0 keras api. For reference here is the original [paper](https://arxiv.org/pdf/1706.02413.pdf) and [code](https://github.com/charlesq34/pointnet2). Where possible I have tried to directly copy and paste original code to avoid discrepancies.

## Setup

Requirements:

* python >= 3.0+
* tensorflow-gpu >= 2.0
* cuda == 10.0

To compile the tensorflow Ops first ensure the `CUDA_ROOT` path in `tf_ops/compile_ops.sh` points correctly to you cuda folder then compile the ops with:

```
chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
```

## Usage

The layers should work as direct replacements for standard `tf.keras.layers` layers, most similarly `Conv2D`. To import just run `from pnet2_layers.layers import <LAYER_NAME>`. To use in your own project just copy the pnet2_layers folder into your project structure and locate with either relative or absolute imports.

For example, to mimic the `pointnet2_cls_ssg` model in the original repository as a custom model, it would look like:

```
import tensorflow
from pnet2_layers.layers import Pointnet_SA

class CLS_SSG_Model(tf.keras.Model)

  def __init__(self, batch_size, activation=tf.nn.relu):
    super(Pointnet2Encoder, self).__init__()

    self.batch_size = batch_size

    self.layer1 = Pointnet_SA(npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128], group_all=False, activation=self.activation)
    self.layer1 = Pointnet_SA(npoint=128, radius=0.4, nsample=32, mlp=[128, 128, 256], group_all=False, activation=self.activation)
    self.layer1 = Pointnet_SA(npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], group_all=False, activation=self.activation)

    # The rest of the model can be implemented using standard tf.keras.layers (Dense and dropout).

  def call():

    xyz, points = self.layer1(input, None)
    xyz, points = self.layer2(xyz, points)
    xyz, points = self.layer3(xyz, points)

    points = tf.reshape(points, (self.batch_size, -1))

    # run points through dense / dropout layers.

    return points
```

Examples of a few of the models from the original repository can be found in the `models` folder.

To run the ModelNet or ScanNet example first download the `tfrecords` containing the training data from [here](https://drive.google.com/open?id=1v5B68RHgDI95KM4EhDrRJxLacJAHcoxz) and place the `train.tfrecord` and `val.tfrecord` files into a folder named either `data/modelnet` or `data/scannet` depending on which dataset you want to train on. To start the training script run either:

```
python train_modelnet.py
```
or:
```
python train_scannet.py
```

If you leave the config to the default values, you can view training logs with:

```
tensorboard --logdir=logs --port=6006
```

and navigate to `localhost:6006` in a web browser.

By default this runs the multi-scale grouping modules. To run the standard set abstraction layer without multi-scale grouping, set `msg` to `False` in the `params` dictionary.

## Notes

I have implemented [batch normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) for all pointnet++ layers in the model files. I personally find I get better results on real world datasets when this option is set to `False`. In the original implementation this batch normalization is set to `True`.

All examples in this repository are using custom training loops and models. This is purely because this is how I use tensorflow. I have not tested these with the high level `Sequential` api, but have no reason to believe it wouldn't work, so if it doesn't, please let me know. I would be happy to add an example model script here if someone makes one.

I will create a model and train script for part segmentation as soon as I find time.
