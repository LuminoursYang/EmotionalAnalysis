"""
Time    :  2020/8/5 16:54 
Author  : Lawrence.Yang
Email   : Lawrence.Yang@connext.com.cn
File    : WordSegmentation.py
Software: PyCharm
"""
import os
import logging
import jieba
import multiprocessing
import configparser
import logging
jieba.setLogLevel(logging.INFO)
config = configparser.ConfigParser()
config.read("NLP.ini")
jieba.load_userdict(config["WordSegmentation"]["public"])
jieba.load_userdict(config["WordSegmentation"]["dongbeihua"])
jieba.load_userdict(config["WordSegmentation"]["makeup"])   # 美妆词包
jieba.load_userdict(config["WordSegmentation"]["babyhealth"])   # 母音词包
jieba.load_userdict(config["WordSegmentation"]["apparel"])  # 鞋服直播
import time



class WordSegmentation():
    """
    用于分词

    单个字符串
    For example:
        ws = WordSegmentation()
        result = ws.cut(sentence)

    字符串list
        ws = WordSegmentation()
        result = ws.batch_cut(sentences_list)
    """
    def __init__(self, n_jobs=1):
        """
        :param n_jobs: int 开启进程数，-1为使用全部核心，默认为1
        """
        # 避免数据类型导致的报错
        try:
            self.n_jobs = int(n_jobs)
        except:
            logging.warning(u"The n_jobs={} was Wrong, change to one core execution by default(n_jobs=1)\n".format(n_jobs))
            self.n_jobs = 1    # 格式化出错，转化为单进程运行


    def cut(self, sentence):
        """单句切分"""
        res = jieba.lcut(sentence)
        return res



    def circle_cut(self, df):
        res = [jieba.lcut(x) for x in df]
        return res



    def res_list_concat(self, list):
        res = []
        for i in list:
            res += i.get()
        return res



    def batch_cut(self, data):
        """
        多进程批量处理
        :data type is list
        :return list
        """
        if len(data) < 700:  # 如果数据量太小则放弃多进程，没必要
            self.n_jobs = 1
        max_cores = os.cpu_count()
        result = []
        if self.n_jobs == -1:
            # 启用全部核心
            pool = multiprocessing.Pool(max_cores)
            for i in range(max_cores):
                if i < max_cores - 1:
                    task = pool.apply_async(
                        self.circle_cut, args=(data[int(i / max_cores * len(data)): int((i + 1) / max_cores * len(data))],))
                else:
                    task = pool.apply_async(
                        self.circle_cut, args=(data[int(i / max_cores * len(data)):],))
                result.append(task.get())
            pool.close()
            pool.join()

        elif (self.n_jobs <= max_cores) | (self.n_jobs > 1):
            # 自定义进程数且小于核心数
            pool = multiprocessing.Pool(self.n_jobs)
            for i in range(self.n_jobs):
                if i < max_cores - 1:
                    task = pool.apply_async(
                        self.circle_cut,
                        args=(data[int(i / self.n_jobs * len(data)): int((i + 1) / self.n_jobs * len(data))],))

                else:
                    task = pool.apply_async(
                        self.circle_cut, args=(data[int(i / self.n_jobs * len(data)):],))
                result.append(task)
            pool.close()
            start = time.time()
            pool.join()
            print("join:", time.time() - start)
        else:
            # 单进程
            result.append(self.circle_cut(data))

        result = self.res_list_concat(result)
        return result







if __name__ == '__main__':
    print(jieba.lcut("片有"))


    exit()
    ws = WordSegmentation(n_jobs= 7)
    sentence = "外观与材质：今天刚刚收到清洁套装，准备送朋友，时间刚刚好。我是你们家的老客户自己孙子使用过一套，非常满意，今天特意表扬你们家客服帅锅莨舰，给我解决在中途出现的一些问题，对他的服务非常满意。"
    print(ws.cut(sentence))
    print(list(jieba.cut(sentence, HMM=False)))

    ws = WordSegmentation(n_jobs= 7)
    start = time.time()
    test = [sentence] * 100000
    test = ws.batch_cut(test)
    print("多进程", time.time() - start)


    ws = WordSegmentation(n_jobs= 1)
    start = time.time()
    test = [sentence] * 100000
    test = ws.batch_cut(test)
    print("单进程", time.time() - start)