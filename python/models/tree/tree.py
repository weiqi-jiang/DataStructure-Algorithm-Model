import math


# 基于ID3 算法的多叉树
class TreeNode:
    def __init__(self,data, label, featureIndex, value, children ):
        # 当前节点上的data
        self.data = data
        #当前节点按照哪一个feature 去分割，记录该feature在当前data中的index
        self.featureIndex = featureIndex 
        # 父节点按照某一个特征分割，当前节点该特征的取值
        self.value = value
        # 子节点 list形式
        self.children = children
        # 当前节点的label值
        self.label = label

class Tree:
    def __init__(self):
        self.root = None 

    def _calcEntropy(self, data):
        labelcount = {}
        data_length = len(data)
        for sample in data:
            label = sample[-1]
            labelcount.setdefault(label, 0)
            labelcount[label] += 1
        entropy = 0.0
        for label in labelcount:
            prob = labelcount[label]/data_length
            entropy -=  prob * math.log2(prob)
        return entropy
    
    def _splitdata(self, data, index, value):
        # 把index对应feature下取值相同的sample聚合在一起
        reducedData = []
        for sample in data:
            if sample[index] == value:
                reducedData.append(sample[:index] + sample[index+1:])
        return reducedData
    
    def _findBestSplit(self,data):
        LargestInfoGain = float('-inf')
        baseEntropy = self._calcEntropy(data)
        splitIndex = -1
        # 遍历除去label的所有特征
        for i in range(len(data[0])-1):
            # 特征的总熵
            feature_entropy = 0.0
            value_set = set([d[i] for d in data])
            # 遍历一个特征下的所有取值
            for value in value_set:
                reducedData = self._splitdata( data, i, value)
                sample_prob = len(reducedData)/ len(data)
                # 特定取值下的熵
                valueEntropy = sample_prob * self._calcEntropy(reducedData)
                feature_entropy += valueEntropy
            
            infogain = baseEntropy - feature_entropy
            if infogain > LargestInfoGain:
                LargestInfoGain = infogain
                splitIndex = i
        return splitIndex

    def _findMajority(self, data):
        # 找到当前多数样本对应的label 作为叶子节点的label
        labelCount = {}
        labels = [d[-1] for d in data]
        for label in labels:
            labelCount.setdefault(label,0)
            labelCount[label] += 1
        sortedlabelcount = sorted(labelCount.items(), key = lambda x: x[1], reverse = True)
        return sortedlabelcount[0][0]

    def buildTree(self, data, root, features):
        features_count = len(features)
        # 样本全为一个类别
        if len(set([d[-1] for d in data])) == 1:
            return TreeNode(data, data[0][-1], None, None, [])
        # 没有可分属性
        if features_count == 0:
            label  = self._findMajority(data)
            return TreeNode(data, label, None, None, [])
        
        bestSplitFeature = self._findBestSplit(data)

        value_set = set([d[bestSplitFeature] for d in data])
        subfeatures = features[:bestSplitFeature] + features[bestSplitFeature+1:]
        for value in value_set:
            newdata = self._splitdata(data, bestSplitFeature, value)
            node = TreeNode(newdata, None, None, value, [] )
            root.featureIndex = bestSplitFeature
            # 递归
            root.children.append(self.buildTree(newdata, node, subfeatures))
        self.root = root
        return root
    
    def showTree(self):
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            print('#############')
            print('Split Feature Index')
            print(current.featureIndex)
            print('parent split value')
            print(current.value)
            print('label')
            print(current.label)
            for child in current.children:
                nodes.append(child)

    def predict(self, data):
        # 预测新输入
        node = self.root
        while node.label is None: 
            splitfeature = node.featureIndex
            for child in node.children:
                if child.label != None:
                    return child.label
                if child.value == data[splitfeature]:
                    node = child
                    data = data[:splitfeature] + data[splitfeature+1:]
                        
        return node.label

    

if __name__ == '__main__':
    tree = Tree()
    data = [
        [2,2,1,0,1],
        [2,2,1,1,0],
        [1,2,1,0,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [1,0,0,1,1],
        [2,1,1,0,0],
        [2,0,0,0,1],
        [0,1,0,0,1],
        [2,1,0,1,1],
        [1,2,0,0,0],
        [0,1,1,1,0]
    ]
    root = TreeNode(data, None, None, None,[])
    root = tree.buildTree(data, root, [i for i in range(len(data[0])-1)])
    tree.showTree()
    print(tree.predict([2,2,1,0]))