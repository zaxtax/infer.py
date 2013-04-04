#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Two Coins
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *
from MicrosoftResearch.Infer.Models import Variable
from MicrosoftResearch.Infer import InferenceEngine

# two coins example
def two_coins() :
    print("\n\n------------------ Infer.NET Two Coins example ------------------\n");
   
    # The model
    b = MicrosoftResearch.Infer.Distributions.Bernoulli(0.5)
    firstCoin = Variable.Bernoulli(0.5)
    secondCoin = Variable.Bernoulli(0.5)
    #bothHeads = firstCoin & secondCoin
    bothHeads = firstCoin.op_BitwiseAnd(firstCoin, secondCoin)

    # The inference
    ie = InferenceEngine()
    print "Probability both coins are heads:", ie.Infer(bothHeads)
    bothHeads.ObservedValue = False
    print "Probability distribution over firstCoin:", ie.Infer(firstCoin)
