import matplotlib.pyplot as plt

def calc_acc(matrix):
    import numpy as np
    diag = np.diag(matrix)
    class_sample_sum = np.sum(matrix, axis=1)
    return diag / class_sample_sum

def score_acc_each_class(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    acc_each_class : numpy.array

    return e.g.:
        array([54.6875    , 85.93981919, 88.63306136, 92.83296542, 87.65483939,
       93.08776717, 99.46170028])
    """
    from sklearn.metrics import confusion_matrix
    acc_each_class = calc_acc(confusion_matrix(y_true, y_pred))*100
    return acc_each_class

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def twotheta_to_d(two_theta, wavelength = 1.54184):
    """
    return d
    """
    import numpy as np
    return wavelength/(2*np.sin(np.deg2rad(two_theta/2)))


def twotheta_to_d_inv(two_theta, wavelength = 1.54184):
    """
    return 1/d
    """
    import numpy as np
    d = wavelength/(2*np.sin(np.deg2rad(two_theta/2)))
    return 1/d


def twotheta_to_d2inv(two_theta, wavelength = 1.54184):
    """
    Parameters

        two_theta: array-like
            list of diffraction peak position in 2theta to convert 1/d^2 notation.
        wavelength: float
            used wavelength in the diffraction experiment.

    Returns
        1/d^2: numpy array
            list of diffraction peak position in 1/d^2 notation.
    """
    import numpy as np
    d = wavelength/(2*np.sin(np.deg2rad(two_theta/2)))
    return 1/(d**2)
