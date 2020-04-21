def froc_curve_tables(df):
	"""
	FROC Curve

	Input:
	------------------------
	Flattened Scores Dataset
	------------------------
	- image idx: index or id of image
	- box idx: index or id of a box in image
	- class_idx: which label is being classified (pedestrian, car, van, cyclist)
	- y_true: 1 or 0 
	- y_prob: prediction probability this rows box is this class
	
	The flattented dataset should look like this:
	The labels and probabilities are not tensors. each row represents the prediction prob. (column 5)
	of a box (column 2) in an image (column 1) being of a class (column 3), along with the true
	boolean value (column 4) of the box being that class.
	
	0  0  0  1  .7
	0  0  1  0  .1
	0  0  2  0  .01
	0  0  3  0  .09
	0  1  0  0  .2
	0  1  1  0  .1
	0  1  2  1  .8
	0  1  3  0  .05

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
	# set up classes
	classes = range(4)
	# threshold range 
	threshold_range = [i/10 for i in range(1, 10, 1)] # currently 1-10, can be changed
	# image index and num boxes per image
	image_indices = range(0, 100)
	# create count structure
	count = {}
	# iterate classes: length of first vector of y_true yields number of classes
	for class_ in classes:
		# init dict
		if not count.get(class_):
			count[class_] = {} 
		# iterate threshold range
		for threshold in threshold_range:
			# init dict
			if not count[class_].get(threshold):
				count[class_][threshold] = {}
			# iterate each image
			for image_idx in image_indices:
				# build count per class, per threshold, per image
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
		# class
		class_ = row[2]
		# index of ground truth label
		y_true_label = row[3]
		# index of predicted label
		y_hat_prob = row[4]
		# get probability, y_hat_label is a class_, and the index of the probability vector
		# iterate thresholds
		for threshold in threshold_range:
			if y_true_label == 1 and y_hat_prob >= threshold:
				# true positives
				count[class_][threshold][image_idx]["true_positives"] += 1
			elif y_true_label == 1 and y_hat_prob < threshold:
				# false negative
				count[class_][threshold][image_idx]["false_negatives"] += 1
			elif y_true_label == 0 and y_hat_prob >= threshold:
				# false positive
				count[class_][threshold][image_idx]["false_positives"] += 1
			elif y_true_label == 0 and y_hat_prob < threshold:
				# true ngative
				count[class_][threshold][image_idx]["true_negatives"] += 1  
			else:
				pass
	###################################
	# calculate average false positives
	###################################
	avg_false_positives = {class_: {} for class_ in classes}
	# store a max average for normalization
	max_avg = 0
	# iterate through count structure to build avg_false_positives per class
	for class_ in count:
		for threshold in count[class_]:
			if not avg_false_positives[class_].get(threshold):
				avg_false_positives[class_][threshold] = 0
			for image_idx in count[class_][threshold]:
				# collapse keys for readability
				c = count[class_][threshold][image_idx]
				# incremental sum to average
				avg_false_positives[class_][threshold] += c["false_positives"] / len(image_indices) 
				# update max average for normalization and binning
				if avg_false_positives[class_][threshold] > max_avg:
					max_avg = avg_false_positives[class_][threshold]
	##########################################
	# aggregate TP and FN values and associate
	# with average false positives
	##########################################
	xy = {class_: {} for class_ in classes}
	# iterate classes
	for class_ in avg_false_positives:
		# iterate thresholds 
		for threshold in avg_false_positives[class_]:
			# get average false positive
			avg_false_positive = avg_false_positives[class_][threshold]
			# get average false positive as a bin value, this rounding can be modified to give a more granular binning 
			avg_false_positive_bin = round(avg_false_positive, 1)
			# associate bin value and corresponding TP/FN with xy data structure per class
			if not xy[class_].get(avg_false_positive_bin):
				xy[class_][avg_false_positive_bin] = {
					"true_positives": c["true_positives"],
					"false_negatives": c["false_negatives"],
				}
			# increase TP/FN with xy per bin value, per class   
			else:
				xy[class_][avg_false_positive_bin]["true_positives"] += c["true_positives"]
				xy[class_][avg_false_positive_bin]["false_negatives"] += c["false_negatives"]
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
	return froc_curve_tables



def test_froc_curve():
	dice = [i/100 for i in range(0, 100, 1)]
	df = []
	for img in range(100):
		for box in range(5):
			for class_ in range(4):
				for ytrue in range(4):
					prob = np.random.choice(dice)
					row = [img, box, class_, ytrue, prob ]
					df.append(row)
	return froc_curve_tables(df)
