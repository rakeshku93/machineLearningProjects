1. In the KMeans class, `the transform() method` measures the distance from each instance to every centroid.
    - Usage of this method: If you have a high-dimensional dataset and you transform it this way, you end up with a k-dimensional dataset.
    - `For example: we had 8-dim X_test data and we reduced it to 3-dim (n_cluster)`
    - bcz, for eacg datapoint `transform method' calculate distance from 3 centriod thus 3-dim
    - this transformation can be a very efficient nonlinear dimensionality reduction technique.
