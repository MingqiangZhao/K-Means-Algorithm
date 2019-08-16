# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:28:09 2019

@author: Zhao Mingqiang
"""

import matplotlib.pyplot as plt
import numpy as np
import math
#from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris


#导入与显示数据集
iris = load_iris()
dataset =  iris.data[:, 2:4]##表示我们只取特征空间中的后两个维度
#绘制数据分布图
plt.scatter(dataset[:, 0], dataset[:, 1], c = "red", marker='o', label='see')  
plt.xlabel('petal length')  
plt.ylabel('petal width')  
plt.legend(loc=2)  
plt.show() 



#定义基本函数
#随机初始化聚类中心
#cluster是一个[K,M]的矩阵，K是聚类中心的个数，M是样本的维度，索引cluster[i:]表示第i类聚类中心的坐标
#dataset是一个[N,M]的矩阵，N是样本的个数，M是样本的维度
def randomInicialize(K,M,dataset):
    
    N = dataset.shape[0]
    cluster = np.zeros((K,M))
    cluster = np.mat(cluster)
    
    for i in range(K):
        index = np.random.randint(N)
        cluster[i,:] = dataset[index,:]
    
    
    return cluster


#距离函数 x1与x2向量的欧几里得距离,x1,x2都表示[1,n]的行向量
def distance(x1,x2):
    #print(x1)
    #print(x2)
    
    if x1.shape != x2.shape:
        print("two vector's shape must the same")
        
    elif (x1.shape[0] != 1) or (x2.shape[0] !=1):
        print("vector shape must like [1,N]")
    else:   
        distance = math.sqrt(np.square(x1-x2).sum(axis=1))
        return distance

#更新所属类别的索引表
#clusterAssemble是一个[N,1]的矩阵，N是样本的个数，记录着每个样本所属的聚类中心，索引clusterAssemble[i:]表示dataset[i:]所属的类
#cluster是一个[K,M]的矩阵，K是聚类中心的个数，M是样本的维度，索引cluster[i:]表示第i类聚类中心的坐标
#dataset是一个[N,M]的矩阵，N是样本的个数，M是样本的维度
def classificationDataset(cluster,dataset):
    K = cluster.shape[0]
    N = dataset.shape[0]
    clusterAssemble = np.zeros((N,1))
    clusterAssemble = np.mat(clusterAssemble)
    
    clusterChanged = False
    for i in range(N):
        MinDis = float("inf")
        MinIndex = -1
        for j in range(K):
          
            dist = distance(dataset[i,:],cluster[j,:])
            if dist < MinDis:
                MinDis = dist
                MinIndex = j
        if clusterAssemble[i,0] != MinIndex: clusterChanged = True
        clusterAssemble[i,0] = MinIndex
    
    return clusterAssemble,clusterChanged



#更新聚类中心
#clusterAssemble是一个[N,1]的矩阵，N是样本的个数，记录着每个样本所属的聚类中心，索引clusterAssemble[i:]表示dataset[i:]所属的类
#cluster是一个[K,M]的矩阵，K是聚类中心的个数，M是样本的维度，索引cluster[i:]表示第i类的坐标
#dataset是一个[N,M]的矩阵，N是样本的个数，M是样本的维度
def update_center(clusterAssemble,cluster,dataset):    
    
    K = cluster.shape[0]
    N = dataset.shape[0]
    M = dataset.shape[1]
    clusterTemp = np.zeros((K,M))
    clusterTemp = np.mat(clusterTemp)
    #print(clusterAssemble)
    for i in range(K):
        count = 0
        for j in range(N):            
            
            #print(clusterAssemble[j,0])
            
            if i == clusterAssemble[j,0]: #找到第j个元素所属的类属于当前的类i
                clusterTemp[i,:] += dataset[j,:]
                count = count + 1
        
        clusterTemp[i,:] = clusterTemp[i,:]/count
    
    cluster = clusterTemp
    return cluster


#定义核心算法，dataset是一个(N,M)的矩阵，N表示样本的个数，M表示样本的维度
def K_means(K,dataset,iteration): 
    
    #N = dataset.shape[0]
    M = dataset.shape[1]
    cluster = randomInicialize(K,M,dataset)
     
    clusterChanged = True
    
    for i in range(iteration):
        clusterAssemble,clusterChanged = classificationDataset(cluster,dataset)
        #print(clusterAssemble)
        cluster = update_center(clusterAssemble,cluster,dataset)
    
    return cluster,clusterAssemble  


dataset = np.mat(dataset)
N = dataset.shape[0]
iteration = 10
    
cluster,clusterAssemble  = K_means(3,dataset,iteration)
#print(cluster)
#print(clusterAssemble)

for i in range(N):
    if clusterAssemble[i,0] == 0:
        plt.scatter(dataset[i,0],dataset[i,1] , c = "red", marker='o', label='see')  
        
    if clusterAssemble[i,0] == 1:
        plt.scatter(dataset[i,0],dataset[i,1] , c = "green", marker='*', label='see')  
        
    if clusterAssemble[i,0] == 2:
        plt.scatter(dataset[i,0],dataset[i,1] , c = "blue", marker='+', label='see')  




'''
x1 = np.array([1,2,3,4,5])
x1 = x1.reshape([1,5])

x2 = np.array([1,2,3,3,4])
x2 = x2.reshape([1,5])

dist = distance(x1,x2)
print(dist)
'''



