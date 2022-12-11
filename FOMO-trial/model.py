import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class FOMO:

    def __init__(self, activation='relu', num_classes=2):
        self.mobilenet = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
        self.cutPoint = self.mobilenet.get_layer('block_6_expand_relu') # cutting mobilenet at 1/8th

        self.head = Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            activation=activation,
            name='head'
        )
        self.logits = Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            activation=None,
            name='logits'
        )

    def headOutput(self):
        x = self.cutPoint.output
        out1 = self.head(x)
        out2 = self.logits(out1)
        return out1, out2

    def modelKeras(self):
        self.output1, self.output2 = self.headOutput()
        return tf.keras.Model(self.mobilenet.inputs, [self.output1, self.output2])

#%%
