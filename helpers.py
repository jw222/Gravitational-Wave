import matplotlib.pyplot as plt
import matplotlib.colors as colors
keyStr = 'data'


def getError(currSess, snr, f, length, shift=None):
    noise = Noiser(length)
    pred = []
    for j in range(len(f[keyStr])):
        test_data = f[keyStr][j].reshape(1, length)
        test_data = noise.add_shift(test_data)
        if shift is not None:
            test_data[0][:shift[0]] = 0
            test_data[0][shift[1]:] = 0
            # test_data[0] = test_data[0][shift[0]:shift[1]]
        if real_noise is False:
            test_data = noise.add_noise(input=test_data, SNR=snr)
        else:
            test_data = noise.add_real_noise(input=test_data, SNR=snr)
        test_data = test_data.reshape(1, length, 1)
        test_label = f['m1m2'][j].reshape(1, 2)
        pred.append(
            currSess.run(predictions,
                         feed_dict={input_data: test_data, input_label: test_label, trainable: False})[0])

    pred = np.asarray(pred)
    test_label = np.asarray(f['m1m2'])
    m1 = np.mean(np.divide(abs(pred.T[0] - test_label.T[0]), test_label.T[0]))
    m2 = np.mean(np.divide(abs(pred.T[1] - test_label.T[1]), test_label.T[1]))
    return m1, m2


def triPlot(predict, name, f):
    testLabel = np.asarray(f['m1m2'])
    error1 = [abs(predict.T[0][n] - testLabel.T[0][n]) / testLabel.T[0][n] for n in range(len(testLabel))]
    error2 = [abs(predict.T[1][n] - testLabel.T[1][n]) / testLabel.T[1][n] for n in range(len(testLabel))]
    plt.figure(figsize=(18, 20))
    cm = plt.cm.get_cmap('seismic')
    plt.subplot(211)
    sc = plt.scatter(testLabel.T[0], testLabel.T[1], c=error1, vmin=0.0025, vmax=0.75,
                     cmap=cm, norm=colors.LogNorm(vmin=np.amin(error1), vmax=np.amax(error1)))
    plt.colorbar(sc)
    plt.xlabel('m1 mass')
    plt.ylabel('m2 mass')
    plt.title(name)
    plt.subplot(212)
    sc = plt.scatter(testLabel.T[0], testLabel.T[1], c=error2, vmin=0.0025, vmax=0.75,
                     cmap=cm, norm=colors.LogNorm(vmin=np.amin(error2), vmax=np.amax(error2)))
    plt.colorbar(sc)
    plt.xlabel('m1 mass')
    plt.ylabel('m2 mass')
    plt.title(name)
    plt.savefig(name + '.png')


def plot(currSess, snrs, f, fig, shift=None):
    # testing without shift
    length = len(f[keyStr][0])
    print("\nshift is: ", shift)
    m1s = []
    m2s = []
    for snr in snrs:
        m1, m2 = getError(currSess, snr, f, length, shift)
        m1s.append(m1)
        m2s.append(m2)
        print('SNR: ' + str(snr) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        if i == 40:
            triPlot(pred, fig + 'triSNR-' + str(snr), f)

    m1s = np.asarray(m1s)
    m2s = np.asarray(m2s)
    plt.figure()
    plt.plot(np.flip(snrs, 0), np.flip(m1s * 100, 0))
    plt.plot(np.flip(snrs, 0), np.flip(m2s * 100, 0))
    plt.legend(['m1', 'm2'], loc=1)
    plt.xlabel('SNR')
    plt.ylabel('Relative Error')
    plt.title('RE with SNR')
    plt.grid(True)
    plt.savefig(fig + '.png')


def gradual(currSess, snrs, f, fig, times):
    length = len(f[keyStr][0])
    num_secs = length // 8192
    for snr in snrs:
        print("\n\nsnr is: ", snr)
        m1s = []
        m2s = []
        for stop in times:
            shift = [0, int(stop * length)]
            m1, m2 = getError(currSess, snr, f, length, shift)
            m1s.append(m1)
            m2s.append(m2)
            print('stop: ' + str(shift) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)

        plt.figure()
        plt.plot(times * num_secs, m1s * 100)
        plt.plot(times * num_secs, m2s * 100)
        plt.legend(['m1', 'm2'], loc=1)
        plt.xlabel('timeStamps in seconds')
        plt.ylabel('Relative Error')
        plt.title('RE with end time')
        plt.grid(True)
        plt.savefig(fig + str(snr) + 'gradual.png')


def window(currSess, snrs, f, fig, step):
    length = len(f[keyStr][0])
    num_secs = length // 8192
    num_shift = (length - 8192) // step + 1
    for snr in snrs:
        print("\n\nsnr is: ", snr)
        m1s = []
        m2s = []
        for i in range(num_shift):
            shift = [i * step, i * step + 8192]
            print("\nwindow is: ", shift)
            m1, m2 = getError(currSess, snr, f, length, shift)
            m1s.append(m1)
            m2s.append(m2)
            print('window: ' + str(shift) + ' -- m1: ' + "{0:.5%}".format(m1) + ' m2: ' + "{0:.5%}".format(m2))
        m1s = np.asarray(m1s)
        m2s = np.asarray(m2s)

        x_axis = np.arange(0, length - 8192 + step, step)
        x_axis /= 8192
        plt.figure()
        plt.plot(x_axis, m1s * 100)
        plt.plot(x_axis, m2s * 100)
        plt.legend(['m1', 'm2'], loc=1)
        plt.xlabel('start of window')
        plt.ylabel('Relative Error')
        plt.title('RE with window')
        plt.grid(True)
        plt.savefig(fig + str(snr) + 'window.png')