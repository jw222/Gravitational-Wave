import tensorflow as tf


class ResidualBlock(tf.keras.Model):
    def __init__(self, dilation_rate, num_filters):
        super(ResidualBlock, self).__init__()
        self.x_f = tf.keras.layers.Conv1D(filters=num_filters,
                                          kernel_size=2,
                                          padding='same',
                                          dilation_rate=dilation_rate,
                                          activation=tf.nn.tanh)
        self.x_g = tf.keras.layers.Conv1D(filters=num_filters,
                                          kernel_size=2,
                                          padding='same',
                                          dilation_rate=dilation_rate,
                                          activation=tf.nn.sigmoid)
        self.multiply = tf.keras.layers.Multiply()
        self.skip = tf.keras.layers.Conv1D(filters=num_filters,
                                           kernel_size=1,
                                           padding='same')
        self.residual = tf.keras.layers.Conv1D(filters=num_filters,
                                               kernel_size=1,
                                               padding='same')
        self.add = tf.keras.layers.Add()

    def call(self, input):
        f = self.x_f(input)
        g = self.x_g(input)
        z = self.multiply([f, g])
        r = self.residual(z)
        s = self.skip(z)
        x = tf.add(r, input)
        return x, s


class WaveNet(tf.keras.Model):
    def __init__(self, dilation_layers=6, num_filters=256):
        super(WaveNet, self).__init__()
        dilation_rates = [2 ** i for i in range(dilation_layers)]
        self.pre = tf.keras.layers.Conv1D(filters=num_filters,
                                          kernel_size=2,
                                          padding='same',
                                          activation=tf.nn.relu)
        self.residual_blocks = [ResidualBlock(dilation_rate, num_filters)
                                for dilation_rate in dilation_rates]
        self.add = tf.keras.layers.Add()
        self.post1 = tf.keras.layers.Conv1D(filters=num_filters//2,
                                            kernel_size=1,
                                            padding='same',
                                            activation=tf.nn.relu)
        self.post2 = tf.keras.layers.Conv1D(filters=1,
                                            kernel_size=1,
                                            padding='same',
                                            activation=tf.nn.sigmoid)
        self.flat = tf.keras.layers.Flatten()

    def call(self, input):
        x = self.pre(input)
        skips = []
        for residual_block in self.residual_blocks:
            x, skip = residual_block(x)
            skips.append(skip)
        out = self.add(skips)
        out = tf.nn.relu(out)
        out = self.post1(out)
        out = self.post2(out)
        out = self.flat(out)
        return out


class TwoChan(tf.keras.Model):
    def __init__(self, model_H, model_L, dilation_layers=6, num_filters=256):
        super(TwoChan, self).__init__()
        self.chanH = WaveNet(dilation_layers, num_filters)
        self.chanH.load_weights(model_H)
        self.chanH.trainable = False
        self.chanL = WaveNet(dilation_layers, num_filters)
        self.chanL.load_weights(model_L)
        self.chanL.trainable = False
        self.conv1 = tf.keras.layers.Conv1D(filters=128,
                                            kernel_size=128,
                                            strides=8,
                                            padding='same',
                                            activation=tf.nn.relu)
        self.up1 = tf.keras.layers.UpSampling1D(size=8)
        self.conv2 = tf.keras.layers.Conv1D(filters=128,
                                            kernel_size=128,
                                            strides=8,
                                            padding='same',
                                            activation=tf.nn.relu)
        self.up2 = tf.keras.layers.UpSampling1D(size=8)
        self.conv3 = tf.keras.layers.Conv1D(filters=256,
                                            kernel_size=256,
                                            strides=8,
                                            padding='same',
                                            activation=tf.nn.relu)
        self.up3 = tf.keras.layers.UpSampling1D(size=8)
        self.conv4 = tf.keras.layers.Conv1D(filters=1,
                                            kernel_size=64,
                                            padding='same',
                                            activation=tf.nn.sigmoid)
        self.flat = tf.keras.layers.Flatten()

    def call(self, input):
        H, L = tf.unstack(input, axis=-1)
        H = tf.expand_dims(H, axis=-1)
        L = tf.expand_dims(L, axis=-1)
        H = self.chanH(H)
        L = self.chanL(L)
        out = tf.stack([H, L], axis=-1)
        out = self.up1(self.conv1(out))
        out = self.up2(self.conv2(out))
        out = self.up3(self.conv3(out))
        out = self.conv4(out)
        out = self.flat(out)
        return out
