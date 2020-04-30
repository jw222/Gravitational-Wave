import tensorflow as tf
import readligo as rl
import sys
import argparse
from net import WaveNet
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
    parser.add_argument('--event', dest='event', type=str, default='150914',
                        help='the event we want to train on')
    parser.add_argument('--channel', dest='channel', type=str, default='H',
                        help='the psd that calculated from that event')
    parser.add_argument('--freq', dest='freq', type=int, default=8,
                        help='frequency used')
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


def generator(file_name, file_prefix, ratio, pSNR=None):
    file_name = file_name.decode('utf-8')
    file_prefix = file_prefix.decode('utf-8')
    with open('psd/' + file_prefix + 'freqs', 'rb') as fh:
        freqs = pickle.load(fh)
    with open('psd/' + file_prefix + 'pxx', 'rb') as fh:
        pxx = pickle.load(fh)
        interp_psd = interp1d(freqs, pxx)

    f_noise = h5py.File('data/' + file_prefix + 'Noise.h5', 'r')
    f_train = h5py.File(file_name, 'r')
    for waveform in f_train['Input']:
        waveform = decimate(waveform, 2)[-8192:]
        if np.random.uniform() < ratio:
            # blank waveform
            whitened = np.zeros(8192)
            target = np.zeros(8192)
        else:
            whitened = whiten(waveform, interp_psd)
            shiftInt = np.random.randint(-1024, 1024)
            whitened = np.roll(whitened, shiftInt)
            if shiftInt >= 0:
                whitened[:shiftInt] = 0
            else:
                whitened[shiftInt:] = 0
            whitened /= np.amax(np.absolute(whitened))
            merger = np.argmax(whitened)
            target = np.zeros(8192)
            target[max(shiftInt, 0):merger] = 1.0
        noiseInt = np.random.randint(20, len(f_noise['noise']) - 8192)
        noise = f_noise['noise'][noiseInt:noiseInt+8192]
        if pSNR is None:
            pSNR = np.random.uniform(0.5, 1.5)
        noise /= np.std(noise)
        output = whitened * pSNR + noise
        output /= np.std(output)
        output = output.reshape(8192, 1)
        yield output, target


if __name__ == '__main__':
    args = parseInput()
    prefix = args.event + args.channel + str(args.freq)
    fileCheck(prefix)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('number of gpus: ', len(gpus))

    if args.save_file:
        stdoutOrigin = sys.stdout
        sys.stdout = open(args.output + '.txt', 'w')

    train_dataset = tf.data.Dataset.from_generator(generator,
                                                   (tf.float64, tf.float64),
                                                   ((8192, 1), 8192),
                                                   (args.train_file, prefix, args.blank_ratio))
    train_dataset = train_dataset.shuffle(buffer_size=9861).batch(args.batch_size)
    validation_dataset = tf.data.Dataset.from_generator(generator,
                                                        (tf.float64, tf.float64),
                                                        ((8192, 1), 8192),
                                                        (args.test_file, prefix, args.blank_ratio))
    validation_dataset = validation_dataset.batch(args.batch_size)

    log_dir = './logs/' + args.output + '/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model = WaveNet(args.num_residuals, args.num_filters)
    tf.keras.backend.clear_session()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy())

    if args.model_path is not None:
        model.load_weights(args.model_path)

    history = model.fit(train_dataset, epochs=args.epoch,
                        validation_data=validation_dataset, validation_freq=args.val_freq,
                        callbacks=[tensorboard_callback])

    model_path = './model/' + args.output
    model.save_weights(model_path, save_format='tf')

    plt.figure(figsize=(15, 8))
    plt.title('loss history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(history.history['loss'])
    plt.plot(np.arange(args.val_freq, args.epoch, args.val_freq), history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig(args.output + '-loss.png')

    # test overall accuracy
    result = []
    snrArr = np.array([3, 2, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    snrArr = np.flip(snrArr)
    for snr in snrArr:
        test_dataset_exist = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((8192, 1), 8192),
                                                            (args.test_file, prefix, 0., snr))
        test_dataset_exist = test_dataset_exist.batch(args.batch_size)
        exist = model.predict(test_dataset_exist)
        test_dataset_blank = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((8192, 1), 8192),
                                                            (args.test_file, prefix, 1., snr))
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
    peaks = testReal(model, prefix[:-1], 8192*10, 4096, args.output, 4096)
    for key in peaks:
        print(key, ': ', peaks[key])
