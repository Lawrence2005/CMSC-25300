# Q3
import numpy as np

# #### Part a #####
# Load in training data and labels
# File available on Canvas
face_data_dict = np. load("face_emotion_data.npz")
face_features = face_data_dict ["X"]
face_labels = face_data_dict ["y"]
n, p = face_features.shape

# Solve the least - squares solution.weights is the array of weight coefficients
weights =(face_features.transpose() @ face_features).inv() @ face_features.transpose() @ face_labels
print(f" Part 4a. Found weights :\n{ weights }")

# Q4
def lstsq_cv_err(features : np.ndarray , labels : np.ndarray ,
subset_count : int =8) -> float :
    """ Estimate the error of a least - squares classifier using cross - validation. Use subset_count different train / test splits with each subset acting as the holdout set once.
    Parameters:
    features(np. ndarray): dataset features as a 2D array with shape(sample_count , feature_count) labels(np. ndarray): dataset class labels(+1/ -1) as a 1D array with length(sample_count)
    subset_count(int): number of subsets to divide the dataset into
    Note : assumes that subset_count divides the dataset evenly
    Returns:
    cls_err(float): estimated classification error rate of least - square method
    """
    sample_count, feature_count = features.shape
    subset_size = sample_count // subset_count
    # Reshape arrays for easier subset - level manipulation
    features = features.reshape(subset_count, subset_size ,
    feature_count)
    labels = labels.reshape(subset_count, subset_size)

    subset_idcs = np.arange(subset_count)
    train_set_size =(subset_count - 1) * subset_size
    subset_err_counts = np.zeros(subset_count)

    for i in range(subset_count):
        train_subset_idcs = subset_idcs [subset_idcs != i]
        X_train = features[train_subset_idcs].reshape(-1, feature_count)
        y_train = labels[train_subset_idcs].flatten()
        X_test = features[i]
        y_test = labels[i]

        w =(X_train.transpose() @ X_train).inv() @ X_train.transpose() @ y_train

        y_pred = np.sign(X_test @ w)
        subset_err_counts[i] = np.sum(y_pred != y_test)
    # Average over the entire dataset to find the classification error
    cls_err = np.sum(subset_err_counts) /(subset_count * subset_size)
    return cls_err
# Run on the dataset with all features included
full_feat_cv_err = lstsq_cv_err(face_features , face_labels)
print(f" Error ␣ estimate :␣{ full_feat_cv_err *100:.3 f}%")

# Q5
import numpy as np
import matplotlib.pyplot as plt

# File available on Canvas
data = np.load('polydata_a24.npz')
x1 = np.ravel(data['x1'])
x2 = np.ravel(data['x2'])
y = data['y']
N = x1.size
p = np.zeros((3, N))
for d in [1 ,2 ,3]:
    # Generate the X matrix for this d, d = degree of the polynomial
    # Find the least-squares weight matrix w_d
    
    # Evaluate the best-fit polynomial at each point(x1 ,x2) and store the result in the corresponding column of p

    # Report the relative error of the polynomial fit
    rel_err = np.linalg.norm(y - <predicted values >) / np.linalg.norm(y )
    print(f"d={d}:␣ relative ␣ error ␣=␣{ rel_err *100:.3 f}%")

# Plot the degree 1 surface
Z1 = p [0 ,:]. reshape(data ['x1 ']. shape )
ax = plt.axes(projection ='3d')
ax.scatter(x1 , x2 , y )
ax.plot_surface(data ['x1 '] , data ['x2 '] , Z1 , color ='orange ')
plt.show()

# Plot the degree 2 surface
Z2 = p [1 ,:]. reshape(data ['x1 ']. shape )
ax = plt.axes(projection ='3d')
ax.scatter(x1 , x2 , y )
ax.plot_surface(data ['x1 '] , data ['x2 '] , Z2 , color ='orange ')
plt.show()

# Plot the degree 3 surface
Z3 = p [2 ,:]. reshape(data ['x1 ']. shape )
ax = plt.axes(projection ='3d')
ax.scatter(x1 , x2 , y )
ax.plot_surface(data ['x1 '] , data ['x2 '] , Z3 , color ='orange ')
plt.show()