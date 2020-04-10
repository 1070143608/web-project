import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neural_network import MLPClassifier
def get_center(origin_data):  # 获取中心点
    '''
    :param origin_data:每个mesh的点数据
    :return: 每个mesh的中心点
    '''
    sum1 = 0
    sum2 = 0
    sum3 = 0
    lo = len(origin_data)
    for i in origin_data:
        sum1 = sum1 + float(i[1])
        sum2 = sum2 + float(i[2])
        sum3 = sum3 + float(i[3])
    av1 = sum1 / lo
    av2 = sum2 / lo
    av3 = sum3 / lo
    return [av1, av2, av3]
def dealData(origin_data):  # origin_data 是一个二维列表, origin是中心点
    '''
    :param origin_data:每个mesh的点数据
    :return: 可用于聚类的二维矩阵
    '''
    origin = get_center(origin_data)
    out_data = []
    for each in origin_data:
        distance = ((float(each[1]) - origin[0]) ** 2 + (float(each[2]) - origin[1]) ** 2 + (
                float(each[3]) - origin[2]) ** 2) ** (1 / 2)
        out_data.append(distance)
    out_array = np.array(out_data).reshape(len(out_data), 1)
    out_array = (out_array - np.sum(out_array) / len(out_array)) * 100
    # print(out_array)
    return out_array  # 输出可用于聚类的数据
def getManyOriginData(path):# 从obj文件夹中提取所有mesh数据
    '''
    :param path:obj文件夹路径
    :return: 三层列表，用于存储每个mesh数据  每个mesh对应的标签列表  每个mesh对应的名称
    '''
    label_dict = {'gem': 1,'gem_fang': 2,'gem_square': 3,
                  'gem_xie': 4, 'other':0}
    label_name = []
    label = []
    two_dimen_list = []
    k = 0
    with open(path, 'r') as f:
        for i in f:
            spl = i.split()
            if spl[0] == 'o':
                if 'gem_fang' in spl[1] or 'gem_square' in spl[1] or 'gem_xie' in spl[1]:
                    label.append(label_dict['gem_fang'])
                elif 'gem' in spl[1]:
                    label.append(label_dict['gem'])
                else:
                    label.append(label_dict['other'])
                label_name.append(spl[1])
                k = len(two_dimen_list)
                two_dimen_list.append([])
                continue
            if spl[0] == 'v':
                two_dimen_list[k].append(spl)
    return two_dimen_list, label, label_name
def cluster(input_data, n_clusters): # 预处理，把mesh数据降维
    km = KMeans(n_clusters=n_clusters)
    label = km.fit_predict(input_data / np.max(input_data))
    output = km.cluster_centers_.reshape(-1)
    output.sort()
    return output / np.max(output)
def getTestData(path): # 获取测试集数据
    '''
    :param path: obj 文件路径
    :return: 测试集数据 测试集标签 训练集名称
    '''
    preMachinLearningData = []
    allData, label, label_name = getManyOriginData(path)
    for i in allData:
        out_array = dealData(i)
        output = cluster(out_array, 20)
        preMachinLearningData.append(list(output))
    train_data = np.array(preMachinLearningData)
    train_label = np.array(label)
    return train_data, train_label, label_name, allData
def getTrainData(path):# 获取训练集数据
    '''
    :param path:  训练集路径
    :return:  训练集数据 训练集标签
    '''
    preMachinLearningData = []
    allData, label, label_name = getManyOriginData(path)
    for i in allData:
        out_array = dealData(i)
        output = cluster(out_array, 20)
        preMachinLearningData.append(list(output))
    train_data = np.array(preMachinLearningData)
    train_label = np.array(label)
    return train_data, train_label
def trainModel(trainDataSet, trainLabel):
        '''训练网络模型'''
        clf = MLPClassifier(hidden_layer_sizes=(100,),
                            activation='relu', solver='adam',
                            learning_rate_init=0.01, max_iter=2000)
        clf.fit(trainDataSet, trainLabel)
        return clf
def predict(testData, testLabel, clf):
    '''预测数据， 检测数据正确率'''
    result = clf.predict(testData)
    error_num = 0
    for i in range(len(testData)):
        if result[i] != testLabel[i]:
            error_num += 1
    return result, error_num / float(len(testData))
class gem(): # 宝石训练模型
    def __init__(self, path):
        self.path = path

    def main(self):
        train_data, train_label = getTrainData(self.path)
        clf = trainModel(train_data, train_label)
        return clf

clf = gem(r'C:\Users\boom\Desktop\xing\machine learning\lib\lib.obj')
trained_model = clf.main()
test_data, test_label, label_name, allData = getTestData(r'C:\Users\boom\Desktop\xing\machine learning\lib\test.obj')
result, error= predict(test_data, test_label, trained_model)
#next_test_data = [allData[i] for i in range(len(result)) if result[i] == 0]

print(result)
print(test_label)
print(error)
nex = []
for i in range(len(result)):
    if result[i] != test_label[i]:
        nex.append(label_name[i])
print(nex)