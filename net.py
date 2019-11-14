import tensorflow as tf


def WaveNet(x, train=True):
    dilation_rates = [2 ** i for i in range(10)]
    receptive_field = sum(dilation_rates) + 2

    # preprocessing causal layer
    x = tf.layers.conv1d(
        inputs=x,
        filters=16,
        kernel_size=2,
        padding="same")

    skips = []

    for dilation_rate in dilation_rates:
        # filter
        x_f = tf.layers.conv1d(
            inputs=x,
            filters=16,
            kernel_size=2,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.tanh)

        # gate
        x_g = tf.layers.conv1d(
            inputs=x,
            filters=16,
            kernel_size=2,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.sigmoid)

        # element wise multiplication
        z = tf.multiply(x_f, x_g)

        # skip cut to account for receptive field
        skip = tf.slice(z, [0, receptive_field, 0], [-1, -1, -1])

        # skip postprocessing
        skip = tf.layers.conv1d(
            inputs=skip,
            filters=32,
            kernel_size=1,
            padding="same")

        # residual postprocessing
        z = tf.layers.conv1d(
            inputs=z,
            filters=16,
            kernel_size=1,
            padding="same")

        # residual connection
        x = tf.add(x, z)

        # skip append
        skips.append(skip)

    # add all skip layers and apply activation
    raw = tf.add_n(skips)
    raw = tf.nn.relu(raw)

    # postprocessing
    raw = tf.layers.conv1d(
        inputs=raw,
        filters=64,
        kernel_size=1,
        padding="same",
        activation=tf.nn.relu)

    # compress to one channel output
    raw = tf.layers.conv1d(
        inputs=raw,
        filters=1,
        kernel_size=1,
        padding="same")

    raw = tf.layers.flatten(raw)

    # get k-highest outputs
    values, indices = tf.nn.top_k(raw, 1024, False)

    m1 = values
    m2 = values
    m1 = tf.layers.dense(m1, units=512, activation=tf.nn.relu)
    m2 = tf.layers.dense(m2, units=512, activation=tf.nn.relu)
    m1 = tf.layers.dropout(inputs=m1, rate=0.25, training=train)
    m2 = tf.layers.dropout(inputs=m2, rate=0.25, training=train)
    m1 = tf.layers.dense(m1, units=256, activation=tf.nn.relu)
    m2 = tf.layers.dense(m2, units=256, activation=tf.nn.relu)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, units=128, activation=tf.nn.relu)
    m2 = tf.layers.dense(m2, units=128, activation=tf.nn.relu)
    m1 = tf.layers.dropout(inputs=m1, rate=0.25, training=train)
    m2 = tf.layers.dropout(inputs=m2, rate=0.25, training=train)
    m1 = tf.layers.dense(m1, units=1, activation=tf.nn.relu)
    m2 = tf.layers.dense(m2, units=1, activation=tf.nn.relu)

    return tf.concat([m1, m2], 1)
