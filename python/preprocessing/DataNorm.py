import numpy as np 
import math

class DataNorm:
    def __init__(self, data):
        self.data = data
        self.min = min(self.data)
        self.max = max(self.data)
        self.avg = np.mean(self.data)
        self.std = np.std(self.data)
        self.sum = sum(self.data)

    def getZScoreNorm(self):
        return [ (d-self.avg)/self.std for d in self.data]

    def getMinMaxNorm(self):
        return [ (d - self.min) / (self.max - self.min) for d in self.data]

    def getDecimalScaling(self):
        j = np.ceil(np.log10(max([abs(d) for d in self.data])))
        return [ d / 10**j for d in self.data]

    def getMeanNorm(self):
        return [ (d - self.avg) / (self.max- self.min) for d in self.data ]
    
    def getVectorNorm(self):
        return [ d / self.sum  for d in self.data]

        

if __name__ == '__main__':
    datanorm = DataNorm([i for i in range(1, 10)])
    print(datanorm.getZScoreNorm())
    print(datanorm.getMeanNorm())
    print(datanorm.getMinMaxNorm())
    print(datanorm.getVectorNorm())
    print(datanorm.getDecimalScaling())