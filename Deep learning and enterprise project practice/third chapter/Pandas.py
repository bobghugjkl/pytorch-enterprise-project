import pandas as pd
import numpy as np

""""Series"""
mylist = list('abced')
myarr = np.arange(5)
ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series([1, 3, 6], index=['a', 'b', 'c'])
print(ser1)
print(ser2)
print(ser3)
print(ser3[['c', 'b']])
"""DataFrame"""
data1 = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
df1 = pd.DataFrame(data1, columns=['Site', 'Age'])
df4 = pd.DataFrame(data1)
print(df1)
print(df4)
data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df2 = pd.DataFrame(data2)
print(df2)
data3 = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}
df3 = pd.DataFrame(data3)
print(df3)
"""loc属性返回指定行数据"""
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45],
}
#数据载入DataFrame对象
df = pd.DataFrame(data)
#返回第一行
print(df.loc[0])
#返回第二行和第三行
print(df.loc[[1, 2]])
"""也可返回列的数据"""
data2 = {
    "mango": [420, 380, 390],
    "apple": [50, 40, 45],
    "pear": [1, 2, 3],
    "banana": [23, 45, 56]
}
df = pd.DataFrame(data2)
print(df[["apple", "banana"]])
