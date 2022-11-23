import tensorflow as tf

class FOMO:

    def __init__(self, activation='relu', num_classes=2):
        self.mobilenet = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
        self.cutPoint = self.mobilenet.get_layer('block_6_expand_relu') # cutting mobilenet at 1/8th

        self.head = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            activation=activation,
            name='head'
        )
        self.logits = tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            activation=None,
            name='logits'
        )

    def headOutput(self):
        x = self.cutPoint.output
        x = self.head(x)
        x = self.logits(x)
        return x

    def modelKeras(self):
        self.output = self.headOutput()
        return tf.keras.Model(self.mobilenet.inputs, self.output)
