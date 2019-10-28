import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import sys
import argparse
from Noiser import Noiser
from Net import WaveNet
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

parser = argparse.ArgumentParser(description='GW code')
parser.add_argument('--model', dest='model_file', type=str, default='test1/model/True_R1noise.ckpt',
                    help='the file of the model')
parser.add_argument('--infer', dest='infer_file', type=str, default='data/oneSecondTest.h5',
                    help='the file of the testing data')
parser.add_argument('--name', dest='name', type=str, default='1on1',
                    help='infer on model')
parser.add_argument('--file', dest='file', type=bool, default=False,
                    help='whether cast output to a file')
parser.add_argument('--noise', dest='real_noise', type=bool, default=True,
					help='whether add real noise or generated noise')
args = parser.parse_args()


model_path = args.model_file
infer_path = args.infer_file
test_num = args.name
real_noise = args.real_noise
if args.file:
    stdoutOrigin=sys.stdout 
    sys.stdout = open("testOut"+test_num+".txt", "w")

f_infer = h5py.File(infer_path, "r")
LENGTH = f_infer['data'].shape[1]

tf.logging.set_verbosity(tf.logging.ERROR)
if np.isnan(f_infer['data']).any():
    print("nan present in training data. Exiting...")
    SystemExit

input_data = tf.placeholder(tf.float32, [None, None, 1])
input_label = tf.placeholder(tf.int32, [None,2])
trainable = tf.placeholder(tf.bool)

# loss function operations
predictions = WaveNet(input_data, trainable)
loss = tf.losses.mean_squared_error(input_label, predictions)

#initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
loss_hist = []
val_loss = []
saver.restore(sess, model_path)


def plot(sess, snrs, f, fig, shift=None):
    def showplot(pred,name):
        test_label = np.asarray(f['m1m2'])
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
    end = LENGTH
    print("\n\nshift is: ", shift)
    noise = Noiser(LENGTH)
    m1s = []
    m2s = []
    for i in range(len(snrs)):
        pred = []
        for j in range(len(f['data'])):
            test_data = f['data'][j][start:end].reshape(1,end-start)
            test_data = noise.add_shift(test_data)
            if shift is not None:
                test_data[0][:shift[0]] = 0
                test_data[0][shift[1]:] = 0
            if real_noise is False:
                test_data = noise.add_noise(input=test_data, SNR=snrs[i])
            else:
                test_data = noise.add_real_noise(input=test_data, SNR=snrs[i])
            test_data = test_data.reshape(1,end-start,1)
            test_label = f['m1m2'][j].reshape(1,2)

            pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])
        pred = np.asarray(pred)
        test_label = np.asarray(f['m1m2'])
        m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
        m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
        m1s.append(m1)
        m2s.append(m2)
        print('SNR: '+str(snrs[i])+' -- m1: '+"{0:.5%}".format(m1)+' m2: '+"{0:.5%}".format(m2))
        #if i % 51 == 0:
        #   showplot(pred,'testSNR--'+fig+str(snrs[i]))

    m1s = np.asarray(m1s)
    m2s = np.asarray(m2s)
    plt.figure()
    plt.plot(np.flip(snrs, 0),np.flip(m1s*100, 0))
    plt.plot(np.flip(snrs, 0),np.flip(m2s*100, 0))
    plt.legend(['m1','m2'], loc=1)
    plt.xlabel('SNR')
    plt.ylabel('Relative Error')
    plt.title('RE with SNR')
    plt.grid(True)
    plt.savefig(fig+'.png')

def gradual(sess, snrs, f, fig, timeStamps):
    noise = Noiser(LENGTH)
    for i in range(len(snrs)):
        print("\n\nsnr is: ", snrs[i])
        m1s = []
        m2s = []
        for stop in timeStamps:
            pred = []
            stop = int(stop*LENGTH)
            print("\n\nstop is: ", stop)
            for j in range(len(f['data'])):
                test_data = f['data'][j].reshape(1,LENGTH)
                test_data = noise.add_shift(test_data)
                test_data[0][stop:] = 0
                if real_noise is False:
                    test_data = noise.add_noise(input=test_data, SNR=snrs[i])
                else:
                    test_data = noise.add_real_noise(input=test_data, SNR=snrs[i])
                test_data = test_data.reshape(1,LENGTH,1)
                test_label = f['m1m2'][j].reshape(1,2)
                pred.append(sess.run(predictions, feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])

            pred = np.asarray(pred)
            test_label = np.asarray(f['m1m2'])
            m1 = np.mean(np.divide(abs(pred.T[0]-test_label.T[0]),test_label.T[0]))
            m2 = np.mean(np.divide(abs(pred.T[1]-test_label.T[1]),test_label.T[1]))
            m1s.append(m1)
            m2s.append(m2)
            print('SNR: '+str(snrs[i])+' -- m1: '+"{0:.5%}".format(m1)+' m2: '+"{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)
        plt.figure()
        plt.plot(timeStamps, m1s*100)
        plt.plot(timeStamps, m2s*100)
        plt.legend(['m1','m2'], loc=1)
        plt.xlabel('timeStamps in seconds')
        plt.ylabel('Relative Error')
        plt.title('RE with input length')
        plt.grid(True)
        plt.savefig(fig+str(snrs[i])+'.png')


snrs = np.linspace(5.0,0.1,50)
plot(sess, snrs, f_infer, 'infer'+test_num+'0.0-1.0s')
plot(sess, snrs, f_infer, 'infer'+test_num+'0.7-0.9s', shift=[int(LENGTH*0.7),int(LENGTH*0.9)])
plot(sess, snrs, f_infer, 'infer'+test_num+'0.5-1.0s', shift=[int(LENGTH*0.5),int(LENGTH*1.0)])

snrs = np.array([5.0,3.0,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1])
timeStamps = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
gradual(sess, snrs, f_infer, 'infer'+test_num+'-', timeStamps)

plot(sess, snrs, f_infer, test_num+'zeroInput', shift=[0,0])

