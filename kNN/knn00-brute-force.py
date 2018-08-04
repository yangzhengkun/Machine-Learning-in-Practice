import numpy as np


def create_dataset():
    # create a matrix: each row as a sample
    group = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels


def kNNClassify(newX, dataset, labels, k=3):
    numsamples = dataset.shape[0]
    # Construct an array by repeating A the number of times given by reps.
    diff = np.tile(A=newX, reps=(numsamples, 1)) - dataset
    square_diff = np.square(diff)
    distance = np.sqrt(np.sum(a=square_diff, axis=1))

    sorted_distance = np.argsort(a=distance)
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    max_count = 0
    max_index = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_index = key
    return max_index


if __name__ == '__main__':
    dataset, labels = create_dataset()
    testX = np.array([1.2, 1.0])
    testY = kNNClassify(testX, dataset, labels, k=3)
    print("output label: {}".format(testY))
