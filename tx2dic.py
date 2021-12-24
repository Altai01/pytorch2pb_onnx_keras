def loadData(filename):
    # dataMat = [];
    labelMat = []
    _dic={}
    fr = open(filename)
    i=1
    for line in fr.readlines():  # 逐行读取
        line = line.strip('\n')
        lineArr = line.strip()  # 滤除行首行尾空格，以\t作为分隔符，对这行进行分解
        data=""
        _dic[i]=lineArr
        i=i+1
        #labelMat.append(lineArr)
    return _dic
    # with open('user_list', 'r', encoding='utf-8') as f:
    #     dic = []
    #     for line in f.readlines():
    #         line = line.strip('\n')  # 去掉换行符\n
    #         b = line.split(' ')  # 将每一行以空格为分隔符转换成列表
    #         dic.append(b)

def save_dict_to_file(_dict, filepath):
    try:
        with open(filepath, 'w') as dict_file:
            for (key,value) in _dict.items():
                dict_file.write('%s:%s\n' % (key, value))
    except IOError as ioerr:
        print ("文件 %s 无法创建" % (filepath))

if __name__ == '__main__' :
    _dict = loadData ('car.txt')
    print(_dict)
    save_dict_to_file(_dict, 'dict_copy.txt')
