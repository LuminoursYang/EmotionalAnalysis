import os
import torch
import pandas as pd
from WordSegmentation import WordSegmentation
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from torch import nn
from configparser import ConfigParser






class Word2vec_torch(nn.Module):
    def __init__(self):
        super(Word2vec_torch, self).__init__()



    def forward(self, x):
        pass





class Word2vec_gensim():
    def __init__(self, path):
        self.sentence_path = path
        self.conf = ConfigParser()
        self.conf.read("NLP.ini")
        with open(self.conf["SentenceProcess"]["punctuation"], encoding="utf8") as f:
            line = f.readlines()
        self.stopwords = [word.rstrip("\n") for word in line]
        self.stopwords.append("\n")
        self.ws = WordSegmentation(n_jobs=7)



    def dataset(self):
        df = pd.read_csv(self.sentence_path, header=None)
        wordlist = df[0].tolist()
        res = self.ws.batch_cut(wordlist)
        res = self.process(res)
        df = pd.DataFrame()
        df[0] = res
        df.to_csv("./data/word2vec.txt", header=None, index=None)



    def process(self, x):
        res = [str([t for t in i if t not in self.stopwords]).replace(",", " ").replace("]", "")
                   .replace("[", "").replace("'", "") for i in x]
        return res



    def training(self, retrain=False):

        # 如果对应路径下没有用于训练word2vec的数据集word2vec.txt，则生成该文件（如果要训练新的记得把老的删了）
        if not os.path.exists("./data/word2vec.txt"):
            self.dataset()

        # 如果对应路下没有word2vec模型参数word2vec.mdl，或retrain设置为True，则开始训练并保存新的模型
        if (not os.path.exists("./model/word2vec.mdl")) | retrain:
            data = LineSentence("./data/word2vec.txt")
            print("训练开始")
            model = Word2Vec(data, window=3, size=100, min_count=5, workers=7, sg=1)    # 若sg=1，则表示训练模型为skip_gram，否则为CBOW
            model.save("./model/word2vec.mdl")






if __name__ == '__main__':
    w2v = Word2vec_gensim("./data/w2v.txt")     # 输入参数为用于训练word2vec的原始数据集所在位置
    w2v.training()

