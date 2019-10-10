import tensorflow as tf

#print(receptive_field)

def WaveNet(x, train=True):
    dilation_rates = [2**i for i in range(10)]
    receptive_field = sum(dilation_rates)+2

    x = tf.layers.batch_normalization(x, training=train)
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
            filters=32,
            kernel_size=2,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.tanh)
        
        # gate
        x_g = tf.layers.conv1d(
            inputs=x,
            filters=32,
            kernel_size=2,
            padding="same",
            dilation_rate=dilation_rate,
            activation=tf.nn.sigmoid)
        
        # element wise multiplication
        z = tf.multiply(x_f,x_g)

        # skip cut to account for receptive field
        skip = tf.slice(z, [0,receptive_field,0], [-1,-1,-1])

        # skip postprocessing
        skip = tf.layers.conv1d(
            inputs=skip,
            filters=128,
            kernel_size=1,
            padding="same")
        
        # residual postprocessing
        z = tf.layers.conv1d(
            inputs=z,
            filters=16,
            kernel_size=1,
            padding="same")
        
        # residual connection
        x = tf.add(x,z)
        
        # skip append
        skips.append(skip)
    
    # add all skip layers and apply activation
    raw = tf.add_n(skips)
    raw = tf.nn.relu(raw)

    # postprocessing
    raw = tf.layers.conv1d(
            inputs=raw,
            filters=128,
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
    #raw = tf.abs(raw)
    values, indices = tf.nn.top_k(raw, 1024, False)
    values = tf.layers.batch_normalization(values, training=train)
    #values = tf.slice(raw, [0,0], [-1,7000])
    
    m1 = values
    m1 = tf.layers.dense(m1, units=512, activation=tf.nn.relu)
    m1 = tf.layers.batch_normalization(m1, training=train)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, units=256, activation=tf.nn.relu)
    m1 = tf.layers.batch_normalization(m1, training=train)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, units=128, activation=tf.nn.relu)
    m1 = tf.layers.batch_normalization(m1, training=train)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, units=1, activation=tf.nn.relu)

    m2 = values
    m2 = tf.layers.dense(m2, units=512, activation=tf.nn.relu)
    m2 = tf.layers.batch_normalization(m2, training=train)
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train) 
    m2 = tf.layers.dense(m2, units=256, activation=tf.nn.relu)
    m2 = tf.layers.batch_normalization(m2, training=train)  
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train) 
    m2 = tf.layers.dense(m2, units=128, activation=tf.nn.relu)
    m2 = tf.layers.batch_normalization(m2, training=train)  
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train) 
    m2 = tf.layers.dense(m2, units=1, activation=tf.nn.relu)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        m1 = tf.identity(m1)
        m2 = tf.identity(m2)

    return tf.concat([m1, m2], 1)


def FixNet(x, train=True):
    x = tf.layers.conv1d(
        inputs=x,
        filters=64,
        kernel_size=16,
        padding="valid")
    x = tf.layers.max_pooling1d(x, 4, 4)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(
        inputs=x,
        filters=128,
        kernel_size=16,
        dilation_rate=2,
        padding="valid")

    x = tf.layers.max_pooling1d(x, 4, 4)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(
        inputs=x,
        filters=256,
        kernel_size=16,
        dilation_rate=2,
        padding="valid")

    x = tf.layers.max_pooling1d(x, 4, 4)
    x = tf.nn.relu(x)
    
    x = tf.layers.conv1d(
        inputs=x,
        filters=512,
        kernel_size=32,
        dilation_rate=2,
        padding="valid")

    x = tf.layers.max_pooling1d(x, 4, 4)
    x = tf.nn.relu(x)

    y = tf.layers.flatten(x)

    y = tf.layers.dense(y, units=128, activation=tf.nn.relu)
    #y = tf.layers.dropout(inputs=y, rate=0.25, training=train)
    y = tf.layers.dense(y, units=64, activation=tf.nn.relu)
    y = tf.layers.dense(y, units=2)

    return y

def FixNet2(input, train=True):
    ratio = 8

    # Three conv1d layers
    def conv1ds(x):
        x = tf.layers.conv1d(
            inputs=x,
            filters=64,
            kernel_size=16,
            dilation_rate=2,
            padding="valid",
            activation=tf.nn.relu)

        x = tf.layers.max_pooling1d(x, 4, 4)

        x = tf.layers.conv1d(
            inputs=x,
            filters=128,
            kernel_size=16,
            dilation_rate=2,
            padding="valid",
            activation=tf.nn.relu)

        x = tf.layers.max_pooling1d(x, 4, 4)

        x = tf.layers.conv1d(
            inputs=x,
            filters=128,
            kernel_size=16,
            dilation_rate=2,
            padding="valid",
            activation=tf.nn.relu)

        x = tf.layers.max_pooling1d(x, 4, 4)

        return x
    
    # SE layers
    def SELayer(x):
        residual = x
        m = tf.reduce_mean(residual, [1], keepdims=True)
        m = tf.layers.dense(m, units=128//ratio, activation=tf.nn.relu)
        m = tf.layers.dense(m, units=128, activation=tf.nn.relu)
        m = m * residual
        residual = residual + m

        m = tf.reduce_mean(residual, [1], keepdims=True)
        m = tf.layers.dense(m, units=128//ratio, activation=tf.nn.relu)
        m = tf.layers.dense(m, units=128, activation=tf.nn.relu)
        m = m * residual
        residual = residual + m

        m = tf.reduce_mean(residual, [1], keepdims=True)
        m = tf.layers.dense(m, units=128//ratio, activation=tf.nn.relu)
        m = tf.layers.dense(m, units=128, activation=tf.nn.relu)
        m = m * residual
        residual = residual + m

        return residual

    # Highway
    def highway(x):
        h = tf.layers.conv1d(
            inputs=x,
            filters=128,
            kernel_size=4,
            padding="same",
            activation=tf.nn.relu)
        t = tf.layers.conv1d(
            inputs=x,
            filters=128,
            kernel_size=4,
            padding="same",
            activation=tf.nn.relu)

        x = h * t + (1 - t) * x

        return x

    # Main layers
    input = conv1ds(input)

    residual1 = input
    residual2 = input
    residual1 = SELayer(residual1)
    residual2 = SELayer(residual2)
    m1 = residual1
    m2 = residual2

    for i in range(30):
        m1 = highway(m1)
        m2 = highway(m2)

    m1 = tf.layers.flatten(m1)
    m1 = tf.layers.dense(m1, 512, activation=tf.nn.relu)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, 256, activation=tf.nn.relu)
    m1 = tf.layers.dropout(inputs=m1, rate=0.1, training=train)
    m1 = tf.layers.dense(m1, 1)

    m2 = tf.layers.flatten(m2)
    m2 = tf.layers.dense(m2, 512, activation=tf.nn.relu)
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train)
    m2 = tf.layers.dense(m2, 256, activation=tf.nn.relu)
    m2 = tf.layers.dropout(inputs=m2, rate=0.1, training=train)
    m2 = tf.layers.dense(m2, 1)

    return tf.concat([m1, m2], 1)



