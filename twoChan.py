import tensorflow as tf
import readligo as rl
import sys
import argparse
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


def generator(file_name, event, ratio, pSNR=None):
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
        if pSNR is None:
            pSNR = np.random.uniform(0.5, 1.5)
        noise_H /= np.std(noise_H)
        noise_L /= np.std(noise_L)
        output_H = whitened_H * pSNR + noise_H
        output_L = whitened_L * pSNR + noise_L
        output_H /= np.std(output_H)
        output_L /= np.std(output_L)
        output = np.stack([output_H, output_L], axis=0)
        output = np.reshape(output, (2, 8192, 1))
        yield output, target


if __name__ == '__main__':
    args = parseInput()
    fileCheck(args.event + 'H8')
    fileCheck(args.event + 'L8')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('number of gpus: ', len(gpus))

    if args.save_file:
        stdoutOrigin = sys.stdout
        sys.stdout = open(args.output + '.txt', 'w')

    train_dataset = tf.data.Dataset.from_generator(generator,
                                                   (tf.float64, tf.float64),
                                                   ((2, 8192, 1), 8192),
                                                   (args.train_file, args.event, args.blank_ratio))
    train_dataset = train_dataset.repeat(2).shuffle(buffer_size=1024).batch(args.batch_size)
    validation_dataset = tf.data.Dataset.from_generator(generator,
                                                        (tf.float64, tf.float64),
                                                        ((2, 8192, 1), 8192),
                                                        (args.test_file, args.event, args.blank_ratio))
    validation_dataset = validation_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

    log_dir = './logs/' + args.output + '/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model = TwoChan(args.model_H, args.model_L, args.num_residuals, args.num_filters)
    tf.keras.backend.clear_session()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.MeanSquaredError())

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
    plt.plot(np.arange(0, args.epoch, args.val_freq), history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig(args.output + '-loss.png')

    # test overall accuracy
    result = []
    snrArr = np.array([3, 2.5, 2, 1.6, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    snrArr = np.flip(snrArr)
    for snr in snrArr:
        test_dataset_exist = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((2, 8192, 1), 8192),
                                                            (args.test_file, args.event, 0., snr))
        test_dataset_exist = test_dataset_exist.batch(args.batch_size)
        exist = model.predict(test_dataset_exist)
        test_dataset_blank = tf.data.Dataset.from_generator(generator,
                                                            (tf.float64, tf.float64),
                                                            ((2, 8192, 1), 8192),
                                                            (args.test_file, args.event, 1., snr))
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
    peaks = testReal(model, event, 8192*10, 4096, args.output, 4096, True)
    for key in peaks:
        print(key, ': ', peaks[key])

