import pandas as pd






if __name__ == '__main__':
    file = "stopwords-cn-all.txt"
    with open(file, "r", encoding="utf8") as f:
        line = f.readlines()
    data = [word.rstrip("\n") for word in line]
    data = pd.DataFrame(data)
    data = data[data[0].apply(lambda x: len(x) == 1)]
    print(data)
    data.to_csv("test.txt", index=None, columns=None)
    # data.to_csv(file, index=None, header=None)