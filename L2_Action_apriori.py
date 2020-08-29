import pandas as pd
import numpy as np
# 数据加载
datas = pd.read_csv('./Market_Basket_Optimisation.csv',header=None)

def rule1(): #采用efficent_apriori求频繁项集和关联规则
	from efficient_apriori import apriori

	# 数据清理，清除空值
	transactions = []
	for i in range(datas.shape[0]):
		data=datas.values[i,:]
		temp_set=data[~pd.isnull(data)]
		transactions.append(temp_set)
    #采用efficent_apriori求频繁项集和关联规则
	itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.2)
	print("频繁项集：", itemsets)
	print("关联规则：", rules)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def rule2():#采用mlxtend.frequent_patterns求频繁项集和关联规则
	from mlxtend.frequent_patterns import apriori
	from mlxtend.frequent_patterns import association_rules

	# 数据加载
	datas = pd.read_csv('./Market_Basket_Optimisation.csv',header=None)

	'''数据整理，对照了breadbasket例子中的data数据格式，将本例中的数据进行了规范化整理。
	由于刚刚学习Python，对pandas的用法非常不熟，用了最笨的方法进行数据的整理，运算速度非常慢，不知道有没有更简单的方法，请老师指导。'''
	new_data=pd.DataFrame(columns=['transactions','items'])#定义
	j=0#记录transactions的行
	for i in range(datas.shape[0]):
		data=datas.values[i,:]
		temp_set=data[~pd.isnull(data)]
		#print(i)
		#print(temp_set)
		for value in temp_set:
			new_data._set_value(j,'transactions',i)
			new_data._set_value(j,'items',value)
			j+=1

	#进行频繁项集和关联规则的求解。
	hot_encoded_df=new_data.groupby(['transactions','items'])['items'].count().unstack().reset_index().fillna(0).set_index('transactions')
	hot_encoded_df = hot_encoded_df.applymap(encode_units)
	frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
	print("频繁项集：", frequent_itemsets)
	print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.2) ])
	print(rules['confidence'])


rule1()
print('-'*100)
rule2()

