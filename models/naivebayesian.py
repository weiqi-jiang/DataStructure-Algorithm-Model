import numpy as np

# P(B|A) = P(A|B)P(B) / P(A) 其中P(A)不需要计算
class NaiveBayesian:
    def __init__(self):
        self.prob_dict  = {}

    def train(self, data):
        labelset = set([d[-1] for d in data])
        for label in labelset:
            label_sample = [d for d in data if d[-1]==label]
            # 计算P(B)
            prob_label = len(label_sample)/len(data)
            self.prob_dict.setdefault(label, [{}, prob_label])
            # 计算label下每个特征的均值和方差
            for index in range(len(label_sample)-1):
                self.prob_dict[label][0].setdefault(index,{})
                sample = [d[index] for d in label_sample]
                avg, std = np.mean(sample), np.std(sample)
                self.prob_dict[label][0][index] = (avg, std)
        return self.prob_dict

    def _calcGaussian(self,mu,sigma,x):
        # 计算高斯分布概率
        return 1.0/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

    def predict(self, data):
        # 输出具有最大后验概率的label
        likelihood_prob = 0
        max_prob = 0.0
        result = None
        for label in self.prob_dict.keys():
            label_prob = self.prob_dict[label][1]
            value_prob_dict = self.prob_dict[label][0]
            for index in range(len(data)):
                std, avg = value_prob_dict[index]
                value_prob = self._calcGaussian(avg, std, data[index])
                label_prob *= value_prob
            if  label_prob > max_prob:
                max_prob = label_prob
                result = label
        return result

            


if __name__ =='__main__':
    '''
    朴素贝叶斯模型用于异常检测，由于数据是连续性，使用高斯平滑，数据最后一位即为label
    '''
    data = [
        [320,204,198,265,1],
        [253,53,15,2243,0],
        [53,32,5,325,0],
        [63,50,42,98,1],
        [1302,523,202,5430,0],
        [32,22,5,143,0],
        [105,85,70,322,1],
        [872,730,840,2762,1],
        [16,15,13,52,1],
        [92,70,21,693,0]
        ]
    naviebayesian = NaiveBayesian()
    print(naviebayesian.train(data))
    print(naviebayesian.predict([134,84,235,349]))