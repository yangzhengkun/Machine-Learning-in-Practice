import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbors

np.random.seed(0)


def create_dataset():
    train_x = np.random.randint(low=0, high=100, size=(25, 2)).astype(np.float32)
    train_y = np.random.randint(low=0, high=2, size=(25, 1)).astype(np.float32)
    return train_x, train_y


def knn_opencv(train_x, train_y, test_x):
    red = train_x[train_y.ravel() == 0]
    plt.scatter(x=red[:, 0], y=red[:, 1], marker='^', c='r')

    blue = train_x[train_y.ravel() == 1]
    plt.scatter(x=blue[:, 0], y=blue[:, 1], marker='s', c='b')

    # 这里要注意：newcommer的数据类型一定要和train_x的数据类型一致np.float32，否则knearest.cpp的312行会报错
    # knearest.cpp line 312：CV_Assert( test_samples.type() == CV_32F && test_samples.cols == samples.cols );
    newcommer = np.float32(test_x)
    plt.scatter(x=newcommer[:, 0], y=newcommer[:, 1], marker='o', c='g')

    knn = cv2.ml.KNearest_create()
    knn.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    ret, results, neighbours, dist = knn.findNearest(samples=newcommer, k=3)
    print("ret:         {}".format(ret))
    print("results:     {}".format(results))
    print("neighbors:   {}".format(neighbours))
    print("dist:        {}".format(dist))
    plt.show()
    return results


def knn_sklearn(train_x, train_y, test_x):
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    train_y = train_y.ravel()
    knn.fit(X=train_x, y=train_y)
    y_pred = knn.predict(X=test_x)
    return y_pred


if __name__ == '__main__':
    train_x, train_y = create_dataset()
    test_x = np.random.randint(low=0, high=100, size=(10, 2)).astype(np.float32)
    opencv_ret = knn_opencv(train_x, train_y, test_x)
    sklearn_ret = knn_sklearn(train_x, train_y, test_x)
    print("opencv_ret:      {}".format(opencv_ret.ravel()))
    print("sklrearn_ret:    {}".format(sklearn_ret))

