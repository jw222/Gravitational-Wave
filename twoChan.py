import tensorflow as tf
import readligo as rl
import sys
import argparse
import time
from net import TwoChan
from utils import *
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parseInput():
    # parsing argument
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--train', dest='train_file', type=str, default='data/TrainEOB_q-1-10-0.02_Flat.h5',
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str, default='data/TestEOB_q-1-10-0.02_Flat.h5',
                        help='the file of the testing data')
    parser.add_argument('--model', dest='model_path', type=str, default=None,
                        help='pretrained model')
    parser.add_argument('--model_H', dest='model_H', type=str, default=None,
                        help='pretrained model of channel H')
    parser.add_argument('--model_L', dest='model_L', type=str, default=None,
                        help='pretrained model of channel L')
    parser.add_argument('--event', dest='event', type=str, default='150914',
                        help='the event we want to train on')
    parser.add_argument('--num_filters', dest='num_filters', type=int, default=256,
                        help='number of filters used in residual blocks')
    parser.add_argument('--num_residuals', dest='num_residuals', type=int, default=6,
                        help='number of residual blocks')
    parser.add_argument('--blank_ratio', dest='blank_ratio', type=float, default=0.5,
                        help='ratio of blank signal used for training and testing')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-4,
                        help='learning rate for training')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--epoch', dest='epoch', type=int, default=100,
                        help='epochs fro training')
    parser.add_argument('--val_freq', dest='val_freq', type=int, default=10,
                        help='validation frequency')
    parser.add_argument('--output', dest='output', type=str, default='test',
                        help='output file name')
    parser.add_argument('--save_file', dest='save_file', type=bool, default=False,
                        help='whether cast output to a file')
    return parser.parse_args()


def generator(file_name, event, ratio, pSNR=[0.5, 1.5]):
    file_name = file_name.decode('utf-8')
    event = event.decode('utf-8')
    with open('psd/' + event + 'H8freqs', 'rb') as fh:
        freqs_H = pickle.load(fh)
    with open('psd/' + event + 'H8pxx', 'rb') as fh:
        pxx_H = pickle.load(fh)
        interp_psd_H = interp1d(freqs_H, pxx_H)
    with open('psd/' + event + 'L8freqs', 'rb') as fh:
        freqs_L = pickle.load(fh)
    with open('psd/' + event + 'L8pxx', 'rb') as fh:
        pxx_L = pickle.load(fh)
        interp_psd_L = interp1d(freqs_L, pxx_L)

    f_noise_H = h5py.File('data/' + event + 'H8Noise.h5', 'r')
    f_noise_L = h5py.File('data/' + event + 'L8Noise.h5', 'r')
    f_train = h5py.File(file_name, 'r')
    for waveform in f_train['Input']:
        waveform = decimate(waveform, 2)[-8192:]
        if np.random.uniform() < ratio:
            # blank waveform
            whitened_H = np.zeros(8192)
            whitened_L = np.zeros(8192)
            target = np.zeros(8192)
        else:
            whitened_H = whiten(waveform, interp_psd_H)
            whitened_L = whiten(waveform, interp_psd_L)
            shiftInt = np.random.randint(-1024, 1024)
            whitened_H = np.roll(whitened_H, shiftInt)
            whitened_L = np.roll(whitened_L, shiftInt)
            if shiftInt >= 0:
                whitened_H[:shiftInt] = 0
                whitened_L[:shiftInt] = 0
            else:
                whitened_H[shiftInt:] = 0
                whitened_H[shiftInt:] = 0
            whitened_H /= np.amax(np.absolute(whitened_H))
            whitened_L /= np.amax(np.absolute(whitened_L))
            merger = int((np.argmax(whitened_H) + np.argmax(whitened_L)) / 2.)
            target = np.zeros(8192)
            target[max(shiftInt, 0):merger] = 1.0
        noiseInt = np.random.randint(20, len(f_noise_H['noise']) - 8192)
        noise_H = f_noise_H['noise'][noiseInt:noiseInt+8192]
        noiseInt = np.random.randint(20, len(f_noise_L['noise']) - 8192)
        noise_L = f_noise_L['noise'][noiseInt:noiseInt+8192]
        snr1 = np.random.uniform(pSNR[0], pSNR[1])
        snr2 = np.random.uniform(pSNR[0], pSNR[1])
        noise_H /= np.std(noise_H)
        noise_L /= np.std(noise_L)
        output_H = whitened_H * snr1 + noise_H
        output_L = whitened_L * snr2 + noise_L
        output_H /= np.std(output_H)
        output_L /= np.std(output_L)
        output = np.stack([output_H, output_L], axis=-1)
        yield output, target


if __name__ == '__main__':
    args = parseInput()
    fileCheck(args.event + 'H8')
    fileCheck(args.event + 'L8')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('number of gpus: ', len(gpus))

    if args.save_file:
        stdoutOrigin = sys.stdout
        sys.stdout = open(args.output + '.txt', 'w')

    while not (os.path.exists(args.model_H + '.index') and os.path.exists(args.model_L + '.index')):
        time.sleep(1800)
    time.sleep(5)

    model = TwoChan(args.model_H, args.model_L, args.num_residuals, args.num_filters)
    if args.model_path is not None:
        model.load_weights(args.model_path)
    model.build((None, 8192, 2))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    criteria = tf.keras.losses.MeanSquaredError()

    losses = []
    maxSNR = np.linspace(1.75, 1.0, args.epoch)
    for epoch in range(args.epoch):
        train_dataset = tf.data.Dataset.from_generator(generator,
                                                       (tf.float64, tf.float64),
                                                       ((8192, 2), 8192),
                                                       (args.train_file, args.event, args.blank_ratio, (0.5, maxSNR[epoch])))
        train_dataset = train_dataset.shuffle(buffer_size=9861).batch(args.batch_size)
        for (batch_n, (input, target)) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(input)
                loss = criteria(target, predictions)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(loss)
            print(f'epoch {epoch} batch {batch_n} has loss: {loss}')

    model_path = './model/' + args.output
    model.save_weights(model_path, save_format='tf')

    plt.figure(figsize=(15, 8))
    plt.title('loss history')
    plt.xlabel('batch number')
    plt.ylabel('loss')
    plt.plot(losses)
    plt.plot(np.poly1d(np.polyfit(range(len(losses)), losses, 5))(range(len(losses))))
    plt.savefig(args.output + '-loss.png')

    # test overall accuracy
    result = []
    snrArr = np.array([3, 2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    snrArr = np.flip(snrArr)
    for snr in snrArr:
        test_dataset_exist = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((8192, 2), 8192),
                                                            (args.test_file, args.event, 0., (snr, snr)))
        test_dataset_exist = test_dataset_exist.batch(args.batch_size)
        exist = model.predict(test_dataset_exist)
        test_dataset_blank = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((8192, 2), 8192),
                                                            (args.test_file, args.event, 1., (snr, snr)))
        test_dataset_blank = test_dataset_blank.batch(args.batch_size)
        blank = model.predict(test_dataset_blank)

        fp = sum([category(pred, 0.5) for pred in blank]) / blank.shape[0]
        sen = sum([category(pred, 0.5) for pred in exist]) / exist.shape[0]
        acc = ((1. - fp) + sen) / 2.
        result.append((fp, sen, acc))

    result = np.asarray(result).T
    plt.figure(figsize=(30, 12))
    plt.title('evaluation v.s. peak snr')
    plt.xlabel('peak snr')
    plt.ylabel('percentage')
    plt.plot(snrArr, result[0])
    plt.plot(snrArr, result[1])
    plt.plot(snrArr, result[2])
    plt.legend(['false_alarm', 'sensitivity', 'accuracy'])
    plt.savefig(args.output + '-evaluation.png')

    # test real signal detection
    peaks = testReal(model, args.event, 8192*10, 4096, args.output, 4096, True)
    for key in peaks:
        print(key, ': ', peaks[key])

