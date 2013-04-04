#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Truncated Gaussian with different thresholds
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *
from MicrosoftResearch.Infer.Models import Variable
from MicrosoftResearch.Infer import InferenceEngine

def truncated_gaussian():
    print("\n\n------------------ Infer.NET Truncated Gaussian example ------------------\n");
    # The model
    threshold = Variable.New[float]().Named("threshold")
    x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x")
    Variable.ConstrainTrue(x > threshold)

    # The inference, looping over different thresholds
    ie = InferenceEngine()
    threshold.ObservedValue = -0.1
    for i in range (0, 11):
        threshold.ObservedValue = threshold.ObservedValue + 0.1
        print "Dist over x given thresh of ", threshold.ObservedValue, "=", ie.Infer(x)
     
