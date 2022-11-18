import numpy as np
import pandas as pd
from IPython.display import display
import distances

class EvaluateImageSegmentation:

	def __init__(self, groundtruth_mask, predicted_mask):
		self.gt_mask = groundtruth_mask
		self.pred_mask = predicted_mask
		self.groundtruth_mask = groundtruth_mask.astype(bool)
		self.predicted_mask = predicted_mask.astype(bool)
		self.intersection = self.groundtruth_mask * self.predicted_mask
		self.union = self.groundtruth_mask + self.predicted_mask
		self.true_positive = self.intersection
		self.false_positive = self.union != self.groundtruth_mask
		self.false_negative = self.union != self.predicted_mask
		self.true_negative = np.invert(self.union)

	def accuracy(self):
		return np.sum(self.true_positive+self.true_negative)/np.sum(self.true_positive+self.true_negative+self.false_negative+self.false_positive)

	def precision(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_positive)

	def recall(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_negative)

	def f1score(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def dice(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def IoU(self):
		return np.sum(self.intersection)/np.sum(self.union)

	def get_confusion_matrix(self):
		gt_series = pd.Series(self.groundtruth_mask.flatten(), name="ground truth")
		pred_series = pd.Series(self.predicted_mask.flatten(), name="predicted")
		df_confusion = pd.crosstab(gt_series, pred_series)
		display(df_confusion)

	def hausdorff_distance(self, distance='euclidean'):
		n1 = self.gt_mask.shape[0]
		n2 = self.pred_mask.shape[0]
		cmax = 0
		for i in range(n1):
			cmin = np.inf
			for j in range(n2):
				dist = getattr(distances, distance)
				dist_cal = dist(self.gt_mask[i,:], self.pred_mask[j,:])
				# dist = np.sqrt(np.sum(np.square(self.gt_mask[i,:] - self.pred_mask[j,:])))
				if dist_cal < cmin:
					cmin = dist_cal
				if cmin < cmax:
					break
			if cmin > cmax and np.inf > cmin:
				cmax = cmin
		return cmax

