#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: clinical trial
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *
from MicrosoftResearch.Infer.Models import Variable, Range
from MicrosoftResearch.Infer import InferenceEngine
from System import Array

def clinical_trial():

    print("\n\n------------------ Infer.NET Clinical Trial example ------------------\n");

    controlGroup = Variable.Observed[Array[bool]](Array[bool]([False, False, True, False, False]))
    treatedGroup = Variable.Observed[Array[bool]](Array[bool]([True, False, True, True, True]))
    i = controlGroup.GetValueRange()
    j = treatedGroup.GetValueRange()

    # Prior on being an effective treatment
    isEffective = Variable.Bernoulli(0.5).Named("isEffective");
    
    # If block
    v = Variable.If(isEffective)
    probIfControl = Variable.Beta(1, 1).Named("probIfControl")
    controlGroup[i] = Variable.Bernoulli(probIfControl).ForEach(i) 
    probIfTreated = Variable.Beta(1, 1).Named("probIfTreated")
    treatedGroup[j] = Variable.Bernoulli(probIfTreated).ForEach(j)
    v.CloseBlock()
    
    # If Not block
    v = Variable.IfNot(isEffective)
    probAll = Variable.Beta(1, 1).Named("probAll")
    controlGroup[i] = Variable.Bernoulli(probAll).ForEach(i)
    treatedGroup[j] = Variable.Bernoulli(probAll).ForEach(j)
    v.CloseBlock()
    
    # The inference
    ie = InferenceEngine()
    print "Probability treatment has an effect = ", ie.Infer(isEffective)
    print "Probability of good outcome if given treatment = ", ie.Infer[Beta](probIfTreated).GetMean()
    print "Probability of good outcome if control = ", ie.Infer[Beta](probIfControl).GetMean()




