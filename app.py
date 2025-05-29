from sklearn.cluster import KMeans 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm

def kmeans_segmentation(image_path, n_clusters=3):
	
	# Load the image
	image = cv2.imread(image_path)
	if image is None:
		raise ValueError("Image not found or unable to read.")

	# Convert the image from BGR to RGB
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Reshape the image to a 2D array of pixels
	pixels = image_rgb.reshape(-1, 3)

	# Apply KMeans clustering
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	kmeans.fit(pixels)

	# Replace each pixel value with its corresponding cluster center
	segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image_rgb.shape).astype(np.uint8)

	return segmented_image

