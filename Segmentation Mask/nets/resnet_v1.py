import tensorflow as tf
import tf_slim as slim
from tensorflow.keras import layers
import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN after convolutions
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that bottleneck variant with extra bottleneck layer is being used here
    When put together two consecutive ResNet blocks , one should use stride = 2 in the last unit of the first block
    Args:
        inputs: A tensor of size [batch, height, width, channels]
        depth: The depth of the ResNet unit output
        depth_bottleneck: The depth of the bottleneck layers
        stride: the ResNet unit's stride. Determines the amount of downsampling of the units output
        compared to its input
        rate: An integer, rate for atrous convolution
        outputs_collections: Collection to add the ResNet unit output
        scope: Optional variable_scope
    Returns:
        The ResNet unit's output"""
    with tf.name_scope(scope or "bottleneck_v1"):
        depth_in = inputs.shape[-1]
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, "shortcut") # Reduce, downsampling along the spatial dimension
        else:
            shortcut = slim.conv2d(inputs, depth, [1,1], stride=stride,
                                   activation_fn=None, scope="shortcut")
        residual = slim.conv2d(inputs, depth_bottleneck, [1,1], stride=1, scope="conv1")
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope = "conv2")
        residual = slim.conv2d(residual, depth, [1,1], stride = 1,
                               activation_fn = None, scope = "conv3")
        output = tf.nn.relu(shortcut + residual)
        return output
def resnet_v1(inputs, blocks, num_classes=None, is_training=True, global_pool = True,
              output_stride = None, include_root_block = True, spatial_squeeze = True, reuse = None,
              scope = None):
    """
    Generator for v1 ResNet models.

    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.

    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.

    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
        Dense prediction meaning predicting a label for each pixel in the image
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    end_points = {}
    with tf.name_scope(scope):
        net = inputs
        # Root block
        if include_root_block:
            if output_stride is not None:
                if output_stride%4 != 0:
                    raise ValueError("The output_stride needs to be a multiple of 4.")
                output_stride /= 4
            net = resnet_utils.conv2d_same(net, 64, 7, stride = 2, scope = "conv1")
            net = slim.max_pool2d(net, [3,3], stride = 2, scope = "pool1")
            end_points["pool2"] = net
        # Stacking blocks
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # Collect intermediate layers
        end_points["pool3"] = end_points['resnet_v1_101/block1']
        end_points["pool4"] = end_points["resnet_v1_101/block2"]
        end_points["pool5"] = net

        # Global pooling (if applicable)
        if global_pool:
            net = tf.reduce_mean(net, axis=[1,2], name="global_pool", keepdims = True)
        if num_classes is not None:
            net = tf.keras.layers.Dense(num_classes, activation = None, name= logits)(net)
            if spatial_squeeze:
                net = tf.squeeze(net, axis=[1,2], name="spatial_squeeze")

        return net, end_points

resnet_v1.default_image_size = 224
def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v1_50'):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v1_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_101'):
    """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_152'):
    """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v1_152.default_image_size = resnet_v1.default_image_size


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_200'):
    """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v1(inputs, blocks, num_classes, is_training,
                     global_pool=global_pool, output_stride=output_stride,
                     include_root_block=True, spatial_squeeze=spatial_squeeze,
                     reuse=reuse, scope=scope)


resnet_v1_200.default_image_size = resnet_v1.default_image_size


if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
    with slim.arg_scope(resnet_arg_scope()) as sc:
        logits = resnet_v1_50(input)