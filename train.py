import datetime
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Noiser import Noiser
from Net import WaveNet
from Batch import get_batch, get_val


f_train = h5py.File("data/TrainEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
f_test = h5py.File("data/TestEOB_q-1-10-0.02_ProperWhitenZ.h5", "r")
tf.logging.set_verbosity(tf.logging.ERROR)

input_data = tf.placeholder(tf.float32, [None, None, 1])
input_label = tf.placeholder(tf.int32, [None,2])
feedlr = tf.placeholder(tf.float32)

# loss function operations
predictions = WaveNet(input_data)
loss = tf.losses.mean_squared_error(input_label, predictions)

# train operation
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.001, 
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
sess = tf.Session()
sess.run(init)
loss_hist = []
val_loss = []
#saver.restore(sess, "../model/shift.ckpt")

num_epoch = 500
start = datetime.datetime.now()
batch_size = 64
rate = 0.001
for i in range(num_epoch):
    train_data, train_label = get_batch(f_train, batch_size)
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
    
    val_data, val_label = get_val(f_test, batch_size)
    validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label})
    val_loss.append(validation)
    print('iter num: '+str(i)+' loss: '+str(loss_hist[-1])+' val_loss: '+str(val_loss[-1]))

end = datetime.datetime.now()
print('time: '+str(end-start))

#save model
save_path = saver.save(sess, "../model/shift.ckpt")
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

pred = []
for i in range(len(f_test['WhitenedSignals'])):
    test_data = f_test['WhitenedSignals'][i][start:end].reshape(1,end-start,1)
    test_label = f_test['m1m2'][i].reshape(1,2)
    pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label})[0])
pred = np.asarray(pred)
test_label = np.asarray(f_test['m1m2'])
m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
print('without shift--m1: '+str(m1)+' m2: '+str(m2))
showplot(pred,'noshift')

pred = []
noise = Noiser()
for i in range(len(f_test['WhitenedSignals'])):
    test_data = f_test['WhitenedSignals'][i][start:end].reshape(1,end-start)
    test_data = noise.add_shift(test_data)
    test_data = test_data.reshape(1,end-start,1)
    test_label = f_test['m1m2'][i].reshape(1,2)
    pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label})[0])
pred = np.asarray(pred)
test_label = np.asarray(f_test['m1m2'])
m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
print('with shift--m1: '+str(m1)+' m2: '+str(m2))
showplot(pred,'shift')