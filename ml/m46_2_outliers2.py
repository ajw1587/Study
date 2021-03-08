# 실습
# outliers1 을 행렬형태도 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
                [100, 200, 3, 400, 500, 600, 700, 8, 900, 1000]])
aaa = aaa.transpose()
print(aaa.shape)                # (10, 2)
print(len(aaa))                 # 10
print(len(aaa[0]))              # 2
print(aaa[:, 0])
print(aaa[:, 1])
print('\n\n')

# data_out2 = np.sort(aaa[:, 1])
# print(data_out2)
# quartile_1, q2, quartile_3 = np.percentile(data_out2, [25, 50, 75])  # 25, 50, 75% 지점의 데이터
# print('1사분위 : ', quartile_1)
# print('q2 : ', q2)
# print('3사분위 : ', quartile_3)
# iqr = quartile_3 - quartile_1
# lower_bound = quartile_1 - (iqr * 1.5)
# upper_bound = quartile_3 + (iqr * 1.5)
# print('lower_bound: ', lower_bound)
# print('upper_bound: ', upper_bound)
# print(np.where((aaa[:, 0] > upper_bound) | (aaa[:, 0] < lower_bound)))


def outliers(data_out):
    for i in range(len(data_out[0])):
        data_out2 = np.sort(data_out[:, i])
        quartile_1, q2, quartile_3 = np.percentile(data_out2, [25, 50, 75])  # 25, 50, 75% 지점의 데이터, 자동 오름차순
        print('1사분위 : ', quartile_1)
        print('q2 : ', q2)
        print('3사분위 : ', quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)
        print(np.where((aaa[:, i] > upper_bound) | (aaa[:, i] < lower_bound)))
        print('\n\n')
    return

outlier_loc = outliers(aaa)



def outliers2(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75], axis = 0, keepdims = True)  # 25, 50, 75% 지점의 데이터
    print('1사분위 : ', quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))


outlier_loc2 = outliers2(aaa)
print(outlier_loc2)

# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()