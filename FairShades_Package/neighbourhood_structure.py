import numpy as np
import pandas as pd

class ClassificationInput(object):
  def generate_input(self, records, labels, neighbourhood, predictions):
    self.records = records
    self.labels = labels
    self.neighbourhood = neighbourhood
    self.predictions = predictions
    orig=[records,labels]
    neighbourhood_predictions=[]
    for i in range(len(neighbourhood)):
      neighbourhood_predictions.append([neighbourhood[i], predictions[i]])
    return orig,neighbourhood_predictions