import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def create_simple_data():
    data_mat = np.matrix([[1., 2.1],
                          [2., 1.1],
                          [1.3, 1.],
                          [1., 1.],
                          [2., 1.]])
    class_labels = np.mat([[1.0], [1.0], [-1.0], [-1.0], [1.0]])
    return data_mat, class_labels


def load_real_samples(filename):
    data = np.loadtxt(fname=filename, dtype=np.float32, delimiter='\t')
    X = data[:, :-1]
    Y = data[:, -1:]
    return X, Y


def stump_classifier(data_mat, dim, threshold, threshold_ineq):
    """
    对训练数据data_mat在第dim特征上按照threshold进行划分
    :param data_mat: 训练数据
    :param dim: 特征索引，第dim维度
    :param threshold:
    :param threshold_ineq: 比较准则，'lt'或'gt'。若为'lt'，则第dim维取值小于threshold的分到类别-1，反之为1
    :return: mx1预测向量
    """
    m, _ = data_mat.shape
    ret_array = np.ones(shape=(m, 1))
    if threshold_ineq == 'lt':
        ret_array[data_mat[:, dim] <= threshold] = -1.0
    else:
        ret_array[data_mat[:, dim] > threshold] = -1.0
    return ret_array


def train_stump(data_mat, class_labels, D):
    """
    训练决策树桩
    :param data_mat: mxn训练样本矩阵，m个样本，每个样本n个特征
    :param class_labels: 样本标签, mx1矩阵
    :param D: 样本权重，mx1矩阵
    :return:
    """
    data_mat = np.array(data_mat)
    class_labels = np.mat(data=class_labels)
    assert data_mat.shape[0] == class_labels.shape[0]
    m, n = data_mat.shape
    best_stump = {}
    best_predict = np.zeros(shape=(m, 1))
    min_error = np.inf
    # 遍历每一个特征
    for i in range(n):
        # 阈值选择：将第i维特征取值升序排序(去除重复)，并取相邻两个值的平均值（也就是区间中点）
        unique_values = np.unique(np.sort(data_mat[:, i]))
        for j in range(len(unique_values) - 1):
            threshold = 0.5 * (unique_values[j] + unique_values[j + 1])
            for ineq in ['lt', 'gt']:
                # 使用决策树桩以threshold作为划分点
                predicted = stump_classifier(data_mat, i, threshold, ineq)
                err_index = np.ones(shape=(m, 1))
                err_index[predicted == class_labels] = 0
                # 错分类样本的权值之和
                weighted_error = np.dot(a=D.transpose(), b=err_index).squeeze()
                # 如果错分率减小，则更新当前最优决策树桩
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_predict = predicted.copy()
                    best_stump["dim"] = i
                    best_stump["threshold"] = threshold
                    best_stump["ineq"] = ineq
    return best_stump, min_error, best_predict


def adaboost_train(data, labels, iters=100):
    """
    训练adaboost分类器
    :param data: 训练数据
    :param labels: 训练数据标签
    :param iters: 迭代次数，也即弱分类器个数
    :return: 弱分类器
    """
    data = np.array(data)
    labels = np.array(labels)
    assert data.shape[0] == labels.shape[0]

    weak_classifiers = []
    m, _ = data.shape

    # 初始样本分布为等权重
    D = np.ones(shape=(m, 1)) / m
    agg_labels_predic = np.zeros(shape=(m, 1))
    for iter in range(iters):
        # 训练决策树桩
        best_stump, min_error, best_predicts = train_stump(data, labels, D)

        # alpha为当前弱分类器的权重
        alpha = np.float(0.5 * np.log((1 - min_error) / max(min_error, 1e-10)))
        alpha = np.array(alpha).squeeze()
        best_stump["alpha"] = alpha
        weak_classifiers.append(best_stump)

        # 更新训练数据集的权值分布
        D = np.multiply(D, np.exp(-alpha * np.multiply(labels, best_predicts)))
        # 对D进行规范化，使其成为一个概率分布
        D /= sum(D)

        # 累加当前及分类的预测值
        agg_labels_predic += alpha * best_predicts
        agg_errors = np.multiply(np.sign(agg_labels_predic) != labels, np.ones((m, 1)))
        print("error_rate:{}".format(agg_errors.sum() / m))
    return weak_classifiers


def adaboost_classifier(data_x, classifiers):
    """
    使用弱分类器进行预测
    :param data_x: 测试数据集
    :param classifiers: 弱分类器，字典
    :return:
    """
    datax = np.mat(data_x)
    m, _ = datax.shape
    agg_predict = np.zeros(shape=(m, 1))
    for classifier in classifiers:
        predicts = stump_classifier(datax, classifier["dim"], classifier["threshold"], classifier["ineq"])
        agg_predict += classifier["alpha"] * predicts
    return np.sign(agg_predict)


if __name__ == '__main__':
    train_file = "horseColicTraining2.txt"
    test_file = "horseColicTest2.txt"
    # data_mat, labels = create_simple_data()
    data_mat, labels = load_real_samples(train_file)
    data_test, labels_test = load_real_samples(test_file)
    m_test, _ = data_test.shape

    m, n = data_mat.shape
    weak_clf = adaboost_train(data_mat, labels, 10)
    predict_test = adaboost_classifier(data_test, weak_clf)
    predict_errors = (predict_test.squeeze() != labels_test.squeeze())
    print("adaboost_error:          ", np.bincount(predict_errors) / m_test)

    # 使用scikit-learn的AdaBoost分类器
    sklearn_adabooster = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                            algorithm="SAMME",
                                            n_estimators=50)
    # scikit-learn的输入y是一向量
    sklearn_adabooster.fit(X=data_mat, y=np.array(labels).squeeze())
    sklearn_predict_test = sklearn_adabooster.predict(data_test)
    sklearn_error_nums = np.bincount(sklearn_predict_test != labels_test.squeeze())
    print("sklearn_adaboost_error:  ", sklearn_error_nums / m_test)
