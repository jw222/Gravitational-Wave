import datetime
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import sys
from Noiser import Noiser
from Net import WaveNet, FixNet, FixNet2
from Batch import get_batch, get_val
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#stdoutOrigin=sys.stdout 
#sys.stdout = open("testOut1.txt", "w")

f_train = h5py.File("data/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
f_test = h5py.File("data/TestEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
tf.logging.set_verbosity(tf.logging.ERROR)

input_data = tf.placeholder(tf.float32, [None, 8192, 1])
input_label = tf.placeholder(tf.int32, [None,2])
#feedlr = tf.placeholder(tf.float32)
trainable = tf.placeholder(tf.bool)

# loss function operations
predictions = FixNet(input_data, trainable)
loss = tf.losses.mean_squared_error(input_label, predictions)

# train operation
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.01, 
										   global_step=global_step, 
										   decay_steps=9861//64, 
										   decay_rate=0.96, 
										   staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(
	loss=loss,
	global_step=global_step)

#initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
loss_hist = []
val_loss = []
#saver.restore(sess, "../model/shift.ckpt")

num_epoch = 500
start = datetime.datetime.now()
batch_size = 64
real_noise = False  #change here!
rate = 0.001
snrs = [3.0,2.0,1.7,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.6,0.5,0.5,0.4,0.4,0.3,0.3,0.3,0.2,0.2,0.2,0.1]
for i in range(num_epoch):
	snr = snrs[i//20]
	train_data, train_label = get_batch(f_train, batch_size, real_noise=real_noise, SNR=snr)
	for j in range(len(train_data)):
		cur_data = train_data[j]
		cur_label = train_label[j]
		_, loss_val = sess.run([train_op, loss],
						   feed_dict={input_data: cur_data,
									  input_label: cur_label,
									  trainable: True})
		loss_hist.append(loss_val)
		if j % 10 == 0:
			print('loss: '+str(loss_hist[-1]))
	
	val_data, val_label = get_val(f_test, batch_size, real_noise=real_noise, SNR=snr)
	validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label, trainable: False})
	val_loss.append(validation)
	print('iter num: '+str(i)+' snr: '+str(snr)+' loss: '+str(loss_hist[-1])+' val_loss: '+str(val_loss[-1]))
	
end = datetime.datetime.now()
print('time: '+str(end-start))

#save model
save_path = saver.save(sess, '../model/test.ckpt', global_step=num_epoch)
print("Model saved in path: %s" % save_path)

step = 9861//batch_size
axis = np.arange(step-1, len(loss_hist), step)
plt.figure()
plt.plot(loss_hist)
plt.scatter(axis, val_loss, c = 'red')
plt.legend(['train_loss','val_loss'], loc=1)
plt.title('loss history--total time: '+str(end-start))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('testLoss.png')


def plot(sess, snrs, f_test, fig, shift=None):
	def showplot(pred,name):
		test_label = np.asarray(f_test['m1m2'])
		error1 = [abs(pred.T[0][i]-test_label.T[0][i])/test_label.T[0][i] for i in range(len(test_label))]
		error2 = [abs(pred.T[1][i]-test_label.T[1][i])/test_label.T[1][i] for i in range(len(test_label))]
		plt.figure(figsize=(18,20))
		cm = plt.cm.get_cmap('seismic')
		plt.subplot(211)
		sc = plt.scatter(test_label.T[0], test_label.T[1], c=error1, vmin=0.0025, vmax=0.75, 
						 cmap=cm, norm=colors.LogNorm(vmin=np.amin(error1), vmax=np.amax(error1)))
		plt.colorbar(sc)
		plt.xlabel('m1 mass')
		plt.ylabel('m2 mass')
		plt.title(name)
		plt.subplot(212)
		sc = plt.scatter(test_label.T[0], test_label.T[1], c=error2, vmin=0.0025, vmax=0.75, 
						 cmap=cm, norm=colors.LogNorm(vmin=np.amin(error2), vmax=np.amax(error2)))
		plt.colorbar(sc)
		plt.xlabel('m1 mass')
		plt.ylabel('m2 mass')
		plt.title(name)
		plt.savefig(name+'.png')

	#testing without shift
	start = 0
	end = 8192

	noise = Noiser()
	m1s = []
	m2s = []
	for i in range(len(snrs)):
		pred = []
		for j in range(len(f_test['WhitenedSignals'])):
			test_data = f_test['WhitenedSignals'][j][start:end].reshape(1,end-start)
			test_data = noise.add_shift(test_data)
			if real_noise is False:
				test_data = noise.add_noise(input=test_data, SNR=snrs[i])
			else:
				test_data = noise.add_real_noise(input=test_data, SNR=snrs[i])
			if shift is not None:
				test_data[:shift[0]] = 0
				test_data[shift[1]:] = 0
			test_data = test_data.reshape(1,end-start,1)
			test_label = f_test['m1m2'][j].reshape(1,2)

			pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])
		pred = np.asarray(pred)
		test_label = np.asarray(f_test['m1m2'])
		m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
		m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
		m1s.append(m1)
		m2s.append(m2)
		print('SNR: '+str(snrs[i])+' -- m1: '+"{0:.5%}".format(m1)+' m2: '+"{0:.5%}".format(m2))
		if i % 50 == 0:
			showplot(pred,'testSNR--'+fig+str(snrs[i]))

	m1s = np.asarray(m1s)
	m2s = np.asarray(m2s)
	plt.figure()
	plt.plot(np.flip(snrs, 0),np.flip(m1s*100, 0))
	plt.plot(np.flip(snrs, 0),np.flip(m2s*100, 0))
	plt.legend(['m1','m2'], loc=1)
	plt.xlabel('SNR')
	plt.ylabel('Relative Error')
	plt.title('RE with SNR')
	plt.savefig(fig+'.png')

snrs = np.linspace(5.0,0.1,249)
#plot(sess, snrs, f_test, '0.7-0.9s', shift=[int(8192*0.7), int(8192*0.9)])
plot(sess, snrs, f_test, '0.0-1.0s')