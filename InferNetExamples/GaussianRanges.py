#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Learning a Gaussian
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)


def gaussian_ranges():
    print
    print
    print "------------------ ",
    print "Infer.NET Learning a Gaussian example",
    print "------------------"
    print

    # The model
    len = Variable.New[int]()
    dataRange = Range(len)
    x = Variable.Array[float](dataRange)
    mean = Variable.GaussianFromMeanAndVariance(0, 100)
    precision = Variable.GammaFromShapeAndScale(1, 1)
    x.set_Item(dataRange,
               Variable.GaussianFromMeanAndPrecision(mean,
                                                     precision).ForEach(dataRange))

    # The data
    data = range(0, 100)  # System.Array.CreateInstance(float, 100)
    for i in range(0, 100):
        data[i] = Rand.Normal(42, 1)

    # Binding the data
    len.ObservedValue = 100
    x.ObservedValue = data

    # The inference
    ie = InferenceEngine(VariationalMessagePassing())
    mean = ie.Infer(mean).GetMean()
    var = ie.Infer(precision).GetMean()

    x = np.linspace(38, 45, 100)
    plt.plot(x, mlab.normpdf(x, mean, var))
    plt.plot(x, mlab.normpdf(x, 42, 1))

    plt.title("Posterior distribution of parameter", fontsize=40)
    plt.xlabel("\mu", fontsize=30)
    plt.show()
