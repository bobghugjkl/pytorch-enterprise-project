import pandas as pd
data = {
    "Date": ['2020/12/01', '2020/12/02', '2020/12/26'],
    "duration": [50, 40, 45]
}
person = {
    "name": ['Google', 'Runoob', 'Taobao'],
    "age": [50, 200, 12345]
}
df1 = pd.DataFrame(data, index=['day1', 'day2', 'day3'])
df1['Date'] = pd.to_datetime(df1['Date'])
print(df1.to_string())
df2 = pd.DataFrame(person)
for x in df2.index:
    if df2.loc[x, 'age'] > 100:
        df2.loc[x, 'age'] = 100
print(df2.to_string())
"""删除重复的数据"""
persons = {
    "name": ['Google', 'Runoob', 'Taobao', 'Taobao'],
    "age": [50, 200, 12345, 12345]
}
df = pd.DataFrame(persons)
df.drop_duplicates(inplace=True)
print(df)