import numpy as np
import pandas as pd
from IPython.display import display


class EvaluateImageSegmentation:

	def __init__(self, groundtruth_mask, predicted_mask):
		self.groundtruth_mask = groundtruth_mask.astype(bool)
		self.predicted_mask = predicted_mask.astype(bool)
		self.intersection = self.groundtruth_mask * self.predicted_mask
		self.union = self.groundtruth_mask + self.predicted_mask
		self.true_positive = self.intersection
		self.false_positive = self.union != self.groundtruth_mask
		self.false_negative = self.union != self.predicted_mask
		self.true_negative = np.invert(self.union)

	def accuracy(self):
		return np.sum(self.true_positive+self.true_negative)/np.sum(self.true_positive+self.true_negative+
																	self.false_negative+self.false_positive)

	def precision(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_positive)

	def recall(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_negative)

	def f1score(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def dice(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def get_confusion_matrix(self):
		gt_series = pd.Series(self.groundtruth_mask.flatten(), name="ground truth")
		pred_series = pd.Series(self.predicted_mask.flatten(), name="predicted")
		df_confusion = pd.crosstab(gt_series, pred_series)
		display(df_confusion)