
import math

class DataDiscrete:
    def __init__(self, data, numBucket):
        self.numBucket = numBucket
        self.data = data

    def _sort(self):
        return sorted(self.data)

    def _calcEntropy(self, data):
        labels = [d[-1] for d in data ]
        labelCounts = {}
        for l in labels:
            labelCounts.setdefault(l, 0)
            labelCounts[l] += 1
        entropy = 0.0
        for count in labelCounts.values():
            prob = count/len(labels)
            entropy += prob * math.log2(prob)
        entropy *= -1
        return entropy



    def getEqualWidth(self):
        data = self._sort()
        diff = max(data) - min(data)
        step = round(diff / self.numBucket, 3)
        left = min(data)
        right = min(left + step, max(data))
        result, bucket = [], 1
        for i in range(len(data)):
            # 等宽边界左闭右开，最大值需要单独处理；如果左开右闭，则最小值需要处理
            if left <= data[i]< right:
                result.append((data[i], bucket))
            else:
                if data[i] == max(data):
                    result.append((data[i], bucket))
                else:
                    left = right
                    right = min(left + step, max(data))
                    bucket += 1
                    result.append((data[i], bucket))
        return result 

    def getEqualCount(self):
        data = self._sort()
        # 如果分桶数大于样本量，则每一个样本一个桶
        if self.numBucket >= len(self.data):
            return [(data[i], i) for i in range(len(data))]
        numForBucket = int(math.ceil(len(self.data)/self.numBucket))
        right = numForBucket
        bucket, result, i = 1, [], 0
        while i< len(data):
            if i >= right:
                right += numForBucket
                bucket += 1
            result.append((data[i], bucket))
            i += 1 
        return result 


    def _getBestSplit(self, data):
        data = sorted(data, key=lambda x: x[0])
        index = -1 
        minEntropy = float('inf')
        # 计算总熵，按照使得总熵最小的index分割
        for i in range(len(data)):
            left, right = data[:i+1], data[i+1:]
            current_entropy = self._calcEntropy(left)*len(left)/len(data) + self._calcEntropy(right)*len(right)/len(data)
            print(current_entropy)
            if current_entropy < minEntropy:
                minEntropy = current_entropy
                index = i 
        return data[:index+1], data[index+1:]


    def getEntropyBased(self):
        toBeSplit = [self.data]
        currentBucket = 1
        # 在没有达到最大分割桶数的时候重复分割，优先分割左边的桶
        while currentBucket < self.numBucket:
            data = toBeSplit.pop(0)
            left, right = self._getBestSplit(data)
            currentBucket += 1
            toBeSplit.extend([left,right])
        result = [(toBeSplit[i], i) for i in range(len(toBeSplit))]
        return result




if __name__ == '__main__':
    # 测试等宽分桶和等频分桶
    datadiscrete = DataDiscrete([1,1,1,1,1,1,2,2,2,2,3,4,5,6,7,8,9,9,9,9,9,9], numBucket=3)
    print(datadiscrete.getEqualWidth())
    print(datadiscrete.getEqualCount())
    # 测试基于信息熵的分桶
    datadiscrete2 = DataDiscrete([(56,1), (87,1), (129,0), (23,0), (342,1)
    ,(641,1),(63,0),(2764,1),(2323,0),(453,1),(10,1),(9,0),(88,1),(222,0),(97,0)
    ,(2398,1),(592,1),(561,1),(764,0),(121,1)], numBucket=3)
    
    for r in datadiscrete2.getEntropyBased():
        print(r)
    