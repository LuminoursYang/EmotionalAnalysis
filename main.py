"""
coding: utf-8
Date  : 2020/9/17 10:31
File  : main.py
Software: PyCharm
Author: Lawrence.Yang
Email: Lawrence.Yang@connext.com.cn
"""
import torch
import pandas as pd
import numpy as np
from torch import nn
from EmtNet_mini import EmtNet
from configparser import ConfigParser
from WordSegmentation import WordSegmentation
from gensim.models import Word2Vec



class EmotionalAnalysis():
    def __init__(self, max_batch=1024):
        self.max_batch = max_batch  # 最大批处理量级
        # 将分词类经行实例化
        self.ws = WordSegmentation()
        # 实例化config文件读取类
        self.conf = ConfigParser()
        # 读取NLP.ini初始化文件
        self.conf.read("NLP.ini")
        # 加载word2vec
        self.w2v = Word2Vec.load(self.conf["Word2vec"]["gensim_path"])
        # 加载停用词模块
        with open(self.conf["SentenceProcess"]["punctuation"], encoding="utf8") as f:
            # 按行读取
            line = f.readlines()
        # 添加停用词
        self.stopwords = [word.rstrip("\n") for word in line]
        # 添加换行符
        self.stopwords.append("\n")
        # 调用搭建模型
        self.net = EmtNet(100)
        # 加载参数
        self.net.load_state_dict(torch.load(r"./model/Emtnet.mdl"))
        self.net.eval()
        # 对应类别中文
        self.res_class = {0: "差评", 1: "中评", 2: "好评"}
        # 判断GPU是否可用
        if torch.cuda.is_available():
            # 可用则调整至GPU模式
            self.device = torch.device('cuda')
        else:
            # 否则调整至CPU模式
            self.device = torch.device('cpu')
        # 调整模型主计算位置
        self.net = self.net.to(self.device)



    def delete_stopwords(self, words_list):
        # 删除停用词
        words_list = [word for word in words_list if word not in self.stopwords]
        return words_list



    def word2vec(self, x):
        res = []
        # 20为最大考虑词汇数，一般对于不会太长的舆情评论20足够了
        for i in range(20):
            if i < len(x):
                res.append(self.embagging(x[i]))
            else:
                res.append([0]*100)
        return res



    def embagging(self, x):
        try:        # 已知词汇初始化
            return list(self.w2v.wv[x])
        except:     # 未知词汇初始化
            return [0] * 100



    def preprocess(self, sentence):
        sentence = self.word2vec(sentence)
        sentence = torch.tensor([sentence], dtype=torch.float32).to(self.device)

        return sentence



    def run(self, sentence):
        sentence = self.ws.cut(sentence)  # 切词
        sentence = self.delete_stopwords(sentence)  # 删除停用词
        sentence = self.preprocess(sentence)

        res = self.net(sentence)
        res = torch.softmax(res, dim=1)[0]
        max = 0
        for i in range(3):
            if res[i] > res[max]:
                max = i
        type = self.res_class[max]
        return type, res



    def res2cn(self, output):
        max = 0
        for i in range(3):
            if output[i] > output[max]:
                max = i
        type = self.res_class[max]
        return type



    def batch_process(self, sentence_list):
        """
        :param sentence_list: ["测试句1", ... "测试局n"]
        :return: ["好评", ... "好评"]
        """
        classification = []     # 分类结果

        wordsegmentation = [self.delete_stopwords(self.ws.cut(sentence)) for sentence in sentence_list]
        sentence_list = [self.preprocess(sentence) for sentence in wordsegmentation]

        times = int(np.ceil(len(sentence_list) / self.max_batch))
        for i in range(times):
            if i < times-1:
                input = torch.cat(sentence_list[i*self.max_batch: (i+1)*self.max_batch])
            else:
                input = torch.cat(sentence_list[i * self.max_batch: ])
            output = self.net(input)
            res = torch.softmax(output, dim=1)

            max_values, max_postion = torch.max(res, dim=1)
            res = [self.res_class[i] for i in max_postion.detach().cpu().tolist()]

            classification += res

        return classification, wordsegmentation








if __name__ == '__main__':
    """   
    该功能已经封装成为一个类，实际使用中最多使用当中两个函数：
        1. 处理单个句子 EmotionalAnalysis.run(string) # 输入字符串
        2. 批量处理多个句子 EmotionalAnalysis.batch_process(list)   # 输入数组，其中每个元素都是字符串
    
    类中参数：
        max_batch: 一次批运算所支持的最大句子数，仅对于批处理函数batch_process生效，调参详情请见./testing
    """
    import time

    EA = EmotionalAnalysis(max_batch=1024)
    s = time.time()
    sentence = [
              "此用户没有填写评论!",
              "太丑了这个口罩，尤其是粉色，居然比黑色大一码，买了没带过，习惯好评吧",
              "感觉还行吧，明天带着试试，应该能防风", "没有赠品！", "有赠品吗？没有！", "有赠品吗？有！",
              "全书跌宕起伏，文笔美不胜收扣人心弦，既不需要很深的文学功底才能理解，又不会过于白话",
              "太棒了",
              "特别棒",
              "赞",
              "不喜欢",
              "喜欢",
              "差劲",
              "不好",
              "好",
              ] * 1
    res, word = EA.batch_process(sentence)
    print(res)
    print((time.time() - s), "s")




