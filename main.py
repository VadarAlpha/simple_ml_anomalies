import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt 
import matplotlib.font_manager 
from pyod.models.knn import KNN  
from pyod.utils.data import generate_data, get_outliers_inliers 

print("Working example of K-Nearest Neighbours from\nhttps://www.geeksforgeeks.org/machine-learning-for-anomaly-detection/")

print("Generating a random dataset with two features...")
X_train, y_train = generate_data(n_train = 300, train_only = True, 
                                                   n_features = 2) 
  

outlier_fraction = 0.1
print("Setting the percentage of outliers: " + str(outlier_fraction))
  
print("Storing the outliers and inliners in different numpy arrays...")
X_outliers, X_inliers = get_outliers_inliers(X_train, y_train) 
n_inliers = len(X_inliers) 
n_outliers = len(X_outliers) 
  
print("Separating the two features...")
f1 = X_train[:, [0]].reshape(-1, 1) 
f2 = X_train[:, [1]].reshape(-1, 1) 


print("Creating a meshgrid for visualizing the dataset...") 
# create a meshgrid 
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), 
                     np.linspace(-10, 10, 200)) 
  
# scatter plot 
plt.scatter(f1, f2) 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 

print("Training the classifier...")
clf = KNN(contamination = outlier_fraction) 
clf.fit(X_train, y_train) 

print("Prediction scores...")  
# You can print this to see all the prediction scores 
scores_pred = clf.decision_function(X_train)*-1



print("Predicting the labels...")  
y_pred = clf.predict(X_train) 
n_errors = (y_pred != y_train).sum() 
# Counting the number of errors 
  
print('The number of prediction errors are ' + str(n_errors)) 

# threshold value to consider a 
# datapoint inlier or outlier 
threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction) 

print("Calculating anomolies...")  
# decision function calculates the raw 
# anomaly score for every point 
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
Z = Z.reshape(xx.shape) 


print("Build the visuals") 
# fill blue colormap from minimum anomaly 
# score to threshold value 
subplot = plt.subplot(1, 2, 1) 
subplot.contourf(xx, yy, Z, levels = np.linspace(Z.min(), 
				threshold, 10), cmap = plt.cm.Blues_r) 

# draw red contour line where anomaly 
# score is equal to threshold 
a = subplot.contour(xx, yy, Z, levels =[threshold], 
					linewidths = 2, colors ='red') 

# fill orange contour lines where range of anomaly 
# score is from threshold to maximum anomaly score 
subplot.contourf(xx, yy, Z, levels =[threshold, Z.max()], colors ='orange') 

# scatter plot of inliers with white dots 
b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], 
									c ='white', s = 20, edgecolor ='k') 

# scatter plot of outliers with black dots 
c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], 
									c ='black', s = 20, edgecolor ='k') 
subplot.axis('tight') 

subplot.legend( 
	[a.collections[0], b, c], 
	['learned decision function', 'true inliers', 'true outliers'], 
	prop = matplotlib.font_manager.FontProperties(size = 10), 
	loc ='lower right') 

print("Displaying the plot...")
subplot.set_title('K-Nearest Neighbours') 
subplot.set_xlim((-10, 10)) 
subplot.set_ylim((-10, 10))
plt.show() 
