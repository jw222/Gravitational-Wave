import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import sys
import os
import argparse
from net import Classifier
from batch import Batch
from inference import Inference
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parseTrainInput():
    # parsing argument
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train', dest='train_file', type=str, default='data/150914H8S1Train.h5',
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str, default='data/150914H8S1Test.h5',
                        help='the file of the testing data')
    parser.add_argument('--noise', dest='noise_file', type=str, default='noise150914-8H1.hdf5',
                        help='the noise file for training')
    parser.add_argument('--name', dest='output_file', type=str, default='1',
                        help='test number')
    parser.add_argument('--keyStr', dest='keyStr', type=str, default='data',
                        help='key to access hdf5 file')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--freq', dest='freq', type=int, default=8,
                        help='sample rate in kHz')
    parser.add_argument('--snr_step', dest='snr_step', type=int, default=5,
                        help='how many steps does each snr train')
    parser.add_argument('--file', dest='file', type=bool, default=False,
                        help='whether cast output to a file')
    parser.add_argument('--noiseType', dest='noiseType', type=bool, default=True,
                        help='whether add real noise or generated noise')
    parser.add_argument('--testOverall', dest='testOverall', type=bool, default=True,
                        help='whether to test overall accuracy')
    parser.add_argument('--testGradual', dest='testGradual', type=bool, default=False,
                        help='whether to test with gradual input')
    parser.add_argument('--testReal', dest='testReal', type=bool, default=False,
                        help='whether to test on real signal')
    return parser.parse_args()


if __name__ == '__main__':
    # initialization
    args = parseTrainInput()
    output_file = args.output_file
    keyStr = args.keyStr
    num_batch = 1

    if args.file:
        stdoutOrigin = sys.stdout
        sys.stdout = open("testOut" + output_file + ".txt", "w")
    tf.logging.set_verbosity(tf.logging.ERROR)

    # set training structure
    input_data = tf.placeholder(tf.float32, [None, None, 1])
    input_label = tf.placeholder(tf.int32, [None, 2])
    trainable = tf.placeholder(tf.bool)
    predictions = Classifier(input_data, trainable)
    loss = tf.losses.softmax_cross_entropy(input_label, predictions)
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)

    # set training initialization
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)

    # training information
    loss_hist = []
    val_loss = []
    new_batch = Batch(args.train_file, args.test_file, args.noise_file, args.batch_size, args.noiseType)
    start_time = datetime.datetime.now()

    low = [0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1]
    snrs = [5.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7] + [lows for lows in low for i in range(3)]
    num_epoch = int(args.snr_step * len(snrs))
    for i in range(num_epoch):
        snrMin = snrs[i // args.snr_step]
        train_data, train_label = new_batch.get_train_batch(snrMin)
        num_batch = len(train_data)
        for j in range(len(train_data)):
            cur_data = train_data[j]
            cur_label = train_label[j]
            _, loss_val = sess.run([train_op, loss],
                                   feed_dict={input_data: cur_data,
                                              input_label: cur_label,
                                              trainable: True})
            loss_hist.append(loss_val)
            if j % 10 == 0:
                print('loss: ' + str(loss_hist[-1]))

        val_data, val_label = new_batch.get_val_batch(snrMin)
        validation = sess.run(loss, feed_dict={input_data: val_data, input_label: val_label, trainable: False})
        val_loss.append(validation)
        print('iter num: ' + str(i) + ' snrMin: ' + str(snrMin) + ' loss: ' 
            + str(loss_hist[-1]) + ' val_loss: ' + str(val_loss[-1]))

    end_time = datetime.datetime.now()
    print('time elapsed: ' + str(end_time - start_time))

    # save model
    model_path = saver.save(sess, '../model/' + output_file + 'Classifier.ckpt')
    print("Model saved in path: %s", model_path)
    val_axis = np.arange(num_batch - 1, len(loss_hist), num_batch)
    plt.figure()
    plt.plot(loss_hist)
    plt.scatter(val_axis, val_loss, c='red')
    plt.legend(['train_loss', 'val_loss'], loc=1)
    plt.title('loss history--total time: ' + str(end_time - start_time))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(output_file + '-testLoss.png')

    # close current session
    sess.close()

    tf.reset_default_graph()
    # testing
    infer = Inference(model_path, args.test_file, args.noise_file, args.freq, args.noiseType, output_file)
    if args.testOverall == True:
        print(args.testOverall)
        #infer.overall_accuracy()
    if args.testGradual:
        infer.gradual_accuracy()
    if args.testReal:
        infer.real_accuracy()
    infer.close()