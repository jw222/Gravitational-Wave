import datetime
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import sys
from Noiser import Noiser
from Net import WaveNet, FixNet
from Batch import get_batch, get_val

f_train = h5py.File("data/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
f_test = h5py.File("data/TestEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
tf.logging.set_verbosity(tf.logging.ERROR)

input_data = tf.placeholder(tf.float32, [None, 8192, 1])
input_label = tf.placeholder(tf.int32, [None,2])
feedlr = tf.placeholder(tf.float32)

# loss function operations
predictions = FixNet(input_data)
loss = tf.losses.mean_squared_error(input_label, predictions)

# train operation
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.001, 
										   global_step=global_step, 
										   decay_steps=9861//64, 
										   decay_rate=0.96, 
										   staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(
	loss=loss,
	global_step=global_step)

#initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
loss_hist = []
val_loss = []
#saver.restore(sess, "../model/shift.ckpt")

num_epoch = 1000
start = datetime.datetime.now()
batch_size = 64
real_noise = False  #change here!
rate = 0.001
snrs = [5.0,4.0,3.0,2.0,1.7,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
for i in range(num_epoch):
	snr = snrs[i//50]
	# global_step.eval(session=sess)
	train_data, train_label = get_batch(f_train, batch_size, real_noise=real_noise, SNR=snr)
	for j in range(len(train_data)):
		cur_data = train_data[j]
		cur_label = train_label[j]
		_, loss_val = sess.run([train_op, loss],
						   feed_dict={input_data: cur_data,
									  input_label: cur_label,
									  feedlr: rate})
		loss_hist.append(loss_val)
		if j % 10 == 0:
			print('loss: '+str(loss_hist[-1]))
	
	val_data, val_label = get_val(f_test, batch_size, real_noise=real_noise, SNR=snr)
	validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label})
	val_loss.append(validation)
	print('iter num: '+str(i)+' snr: '+str(snr)+' loss: '+str(loss_hist[-1])+' val_loss: '+str(val_loss[-1]))
	
end = datetime.datetime.now()
print('time: '+str(end-start))

#save model
save_path = saver.save(sess, '../model/test.ckpt', global_step=num_epoch)
print("Model saved in path: %s" % save_path)

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
snr = np.linspace(3.0,0.1,300)
m1s = []
m2s = []
for i in range(len(snr)):
	pred = []
	for j in range(len(f_test['WhitenedSignals'])):
		test_data = f_test['WhitenedSignals'][j][start:end].reshape(1,end-start)
		test_data = noise.add_shift(test_data)
		if real_noise is False:
				test_data = noise.add_noise(input=test_data, SNR=snr[i])
		else:
			test_data = noise.add_real_noise(input=test_data, SNR=snr[i])
		test_data = test_data.reshape(1,end-start,1)
		test_label = f_test['m1m2'][j].reshape(1,2)
		pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label})[0])
	pred = np.asarray(pred)
	test_label = np.asarray(f_test['m1m2'])
	m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
	m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
	m1s.append(m1)
	m2s.append(m2)
	print('SNR: '+str(snr[i])+' -- m1: '+"{0:.5%}".format(m1)+' m2: '+"{0:.5%}".format(m2))
	if i % 50 == 0:
		showplot(pred,'testSNR--'+str(snr[i]))

m1s = np.asarray(m1s)
m2s = np.asarray(m2s)
plt.figure()
plt.plot(np.flip(snr, 0),m1s*100)
plt.plot(np.flip(snr, 0),m2s*100)
plt.legend(['m1','m2'], loc=1)
plt.xlabel('SNR')
plt.ylabel('Relative Error')
plt.title('RE with SNR')
plt.savefig('testSNR.png')
