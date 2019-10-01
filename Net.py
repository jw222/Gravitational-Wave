import tensorflow as tf

#print(receptive_field)

def WaveNet(x, train=True):
	dilation_rates = [2**i for i in range(10)]
	receptive_field = sum(dilation_rates)+2
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
			padding="causal",
			dilation_rate=dilation_rate,
			activation=tf.nn.tanh)
		
		# gate
		x_g = tf.layers.conv1d(
			inputs=x,
			filters=32,
			kernel_size=2,
			padding="causal",
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
	
	# max pooling to get k outputs
	#raw = tf.layers.max_pooling1d(raw, (LENGTH-receptive_field)//512, (LENGTH-receptive_field)//512)
	#raw = tf.slice(raw, [0,0,0], [-1,512,-1])
	'''
	values = tf.layers.conv1d(
			inputs=raw,
			filters=16,
			kernel_size=8,
			padding="same",
			activation=tf.nn.relu)
	values = tf.layers.max_pooling1d(values, 4, 4)
	
	
	values = tf.layers.conv1d(
			inputs=values,
			filters=1,
			kernel_size=1,
			padding="same")
	'''	
	values = tf.layers.flatten(raw)
	
	# get k-highest outputs
	#values, indices = tf.nn.top_k(values, 1024, False)
	values = tf.slice(values, [0,0], [-1,7000])
	
	values = tf.layers.dense(values, units=1024, activation=tf.nn.relu)
	if train is True:
		values = tf.layers.dropout(inputs=values, rate=0.15)
	values = tf.layers.dense(values, units=128, activation=tf.nn.relu)
	if train is True:
		values = tf.layers.dropout(inputs=values, rate=0.15)

	#indices = tf.divide(tf.cast(indices, tf.float32), tf.cast(length, tf.float32))
	#indices = tf.layers.dense(indices, units=128, activation=tf.nn.relu)
	#indices = tf.layers.dropout(inputs=indices, rate=0.1)
	#indices = tf.layers.dense(indices, units=128, activation=tf.nn.relu)
	#indices = tf.layers.dropout(inputs=indices, rate=0.1)
	
	#out = tf.multiply(tf.nn.sigmoid(indices), tf.nn.tanh(values))

	out = tf.layers.dense(values, units=2)

	return out


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
	#if train is True:
	#	y = tf.layers.dropout(inputs=y, rate=0.25)
	y = tf.layers.dense(y, units=64, activation=tf.nn.relu)
	y = tf.layers.dense(y, units=2)

	return y


