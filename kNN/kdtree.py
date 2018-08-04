import numpy as np


class KDNode(object):
    def __init__(self, data, split, left, right):
        self.data = data  # k维数据样本，列向量
        self.split = split  # 切分维度索引index
        self.left = left  # 左子树
        self.right = right  # 右子树


class KDTree(object):
    def __init__(self, dataset):
        # 转换dataset为numpy数组，每一行对应一个样本
        dataset = np.array(dataset)
        nsamples, ndim = dataset.shape

        # 按第split维划分dataset
        def create_node(split, data_set):
            if not data_set.any():
                return None
            # 按第spilt维对数据排序
            data_set = data_set[data_set[:, split].argsort()]
            split_pos = len(data_set) // 2
            median = data_set[split_pos]  # 中位数
            split_next = (split + 1) % ndim
            return KDNode(median, split, create_node(split_next, data_set[:split_pos]),
                          create_node(split_next, data_set[split_pos + 1:]))

        self.root = create_node(0, dataset)


def preorder(root):
    if root is None:
        return
    print(root.data)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


def inorder(root):
    if root is None:
        return
    if root.left:
        inorder(root.left)
    print(root.data)
    if root.right:
        inorder(root.right)


if __name__ == "__main__":
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kd = KDTree(data)
    print("preorder:")
    preorder(kd.root)
    print("\ninorder: ")
    inorder(kd.root)
