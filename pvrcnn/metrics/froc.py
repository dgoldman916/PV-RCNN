try:
	from pvrcnn.core import cfg
except:
	class C:
		def __init__(self):
			self.NUM_CLASSES = 3

	cfg = C()

import numpy as np


def froc_curve_tables(df):
	"""
	FROC Curve

	Input:
	------------------------
	Flattened Scores Dataset
	------------------------
	- image idx: index or id of image
	- box idx: index or id of a box in image
	- y_hat: which label is being classified (pedestrian, car, van, cyclist)
	- y_true: 1 or 0 
	- y_prob: prediction probability this rows box is this class
	
	The flattented dataset should look like this:
	The labels and probabilities are not tensors. each row represents the prediction prob. (column 5)
	of a box (column 2) in an image (column 1) being of a class (column 3), along with the true
	boolean value (column 4) of the box being that class.

	0  0  0  0  .7
	0  1  1  1  .1
	0  2  1  0  .01

	ADD IN FALSE POSITIVES THAT ARE ACTUALLY NaN

	0  3  0  1  .09
	0  4  1  1  .2
	1  0  1  0  .1
	1  1  0  0  .8
	1  2  1  0  .05

	Output:
	-----------------
	FROC Curve Tables
	-----------------
	Per class: one vs. all?
	- x: average # of False Positives per image
	- y: Sensitivity: the rate of True Positives / (True Positives + False Negatives) per class at x.

	Given a certain threshold, False Positives per image will change.
	""" 
	####################
	# set up count table
	####################
	classes = range(cfg.NUM_CLASSES)
	# threshold range currently 1-10, can be changed?
	threshold_range = [i/10 for i in range(1, 10, 1)]
	image_indices = sorted(list(set(([row[0] for row in df]))))
	count = {}
	for class_ in classes:
		if not count.get(class_):
			count[class_] = {} 
		for threshold in threshold_range:
			# build count per class, per threshold, per image
			if not count[class_].get(threshold):
				count[class_][threshold] = {}
			for image_idx in image_indices:
				count[class_][threshold][image_idx] = {
					# "num_proposals"   : num_proposals_per_image[idx], # number of prediction bounding boxes
					"false_positives" : 0, 
					"true_positives"  : 0,
					"false_negatives" : 0,
					"true_negatives"  : 0  
				}
	##################
	# count TP, FP, FN
	##################
	for row in df:
		image_idx = row[0]
		class_ = row[2]
		matches_y_true = row[3]
		y_hat_prob = row[4]
		# get probability, y_hat_label is a class_, and the index of the probability vector
		for threshold in threshold_range:
			if matches_y_true == 1 and y_hat_prob >= threshold:
				count[class_][threshold][image_idx]["true_positives"] += 1
			elif matches_y_true == 1 and y_hat_prob < threshold:
				count[class_][threshold][image_idx]["false_negatives"] += 1
			elif matches_y_true == 0 and y_hat_prob >= threshold:
				count[class_][threshold][image_idx]["false_positives"] += 1
			elif matches_y_true == 0 and y_hat_prob < threshold:
				count[class_][threshold][image_idx]["true_negatives"] += 1  
			else:
				pass
	###################################
	# calculate average false positives
	###################################
	avg_false_positives = {class_: {} for class_ in classes}
	# iterate through count structure to build avg_false_positives per class
	for class_ in count:
		for threshold in count[class_]:
			# increment fp
			if not avg_false_positives[class_].get(threshold):
				avg_false_positives[class_][threshold] = 0
			for image_idx in count[class_][threshold]:
				avg_false_positives[class_][threshold] += count[class_][threshold][image_idx]["false_positives"]
			# avg fp
			avg_false_positives[class_][threshold] /= len(image_indices)
	##########################################
	# aggregate TP and FN values and associate
	# with average false positives
	##########################################
	xy = {class_: {} for class_ in classes}
	for class_ in avg_false_positives:
		# iterate thresholds 
		for threshold in avg_false_positives[class_]:
			image_metrics_by_threshold = count[class_][threshold]
			# get average false positive
			avg_false_positive = avg_false_positives[class_][threshold]
			# get average false positive as a bin value, this rounding can be modified to give a more granular binning 
			avg_false_positive_bin = round(avg_false_positive, 1)
			for idx, image_metrics in image_metrics_by_threshold.items():
				# associate bin value and corresponding TP/FN with xy data structure per class
				if not xy[class_].get(avg_false_positive_bin):
					xy[class_][avg_false_positive_bin] = {
						"true_positives": image_metrics["true_positives"],
						"false_negatives": image_metrics["false_negatives"],
					}
				# increase TP/FN with xy per bin value, per class   
				else:
					xy[class_][avg_false_positive_bin]["true_positives"] += image_metrics["true_positives"]
					xy[class_][avg_false_positive_bin]["false_negatives"] += image_metrics["false_negatives"]
	#########################
	# build froc curve tables
	#########################
	froc_curve_tables = {class_: {} for class_ in classes}
	for class_ in xy:
		for avg_false_positive_bin in xy[class_]:
			# collapse keys for readability
			c = xy[class_][avg_false_positive_bin]
			# calculate sensitivity
			sensitivity = c["true_positives"] / (c["true_positives"] + c["false_negatives"])
			# key is x-axis: avg false positives per image, value is y-axis sensitivity per avg false positives per image
			froc_curve_tables[class_][avg_false_positive_bin] = sensitivity
	return froc_curve_tables, avg_false_positives, count, xy



def test_froc_curve(y_hat_prob_dist=None, y_true_prob_dist=None, y_prob_prob_dist=None, boxes_per_image_prob_dist=None, boxes_per_image_max=10, num_images=100):
	"""
	Tests the froc curve function using a generated dataset of predictions and truth values.
	The data for testing the froc curve can be randomly generated, and modulated using the 
	kwargs that control the probability distributions for randomly sampling.
	"""
	prob_range = np.array([i/100 for i in range(0, 100, 1)])
	# probabilty distributions for randomly generating y_hat/y_true labels default to None (randomness)
	#for prob_dist in [y_hat_prob_dist, y_true_prob_dist, y_prob_prob_dist, boxes_per_image_prob_dist]:
		#if isinstance(prob_dist, np.ndarray):
			# probability scores are 100 values between 0 and 1
		#	if prob_dist == y_prob_prob_dist:
		#		assert len(prob_dist) == 100
			# probability distribution of image having up to a certain number of boxes
		#	elif prob_dist == boxes_per_image_prob_dist:
		#		assert len(prob_dist) == boxes_per_image_max
			# y_hat or y_true probability distributions
		#	else:
		#		assert len(prob_dist) == cfg.NUM_CLASSES
		#	assert isinstance(prob_dist, np.ndarray)
		#	assert np.sum(prob_dist) == 1.0
	# generate dataset
	df = []
	for img in range(num_images):
		num_boxes = np.random.choice(range(boxes_per_image_max), p=boxes_per_image_prob_dist)
		for box in range(num_boxes):
			# set y_hat, y_true, y_prob
			y_hat = np.random.choice(range(cfg.NUM_CLASSES), p=y_hat_prob_dist)
			y_true = np.random.choice(range(cfg.NUM_CLASSES), p=y_true_prob_dist)
			if y_true == y_hat:
				y_true = 1
			else:
				y_true = 0
			# uses prob range
			y_prob = np.random.choice(prob_range, p=y_prob_prob_dist)
			row = [img, box, y_hat, y_true, y_prob]
			df.append(row)
	return froc_curve_tables(df)
