import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#input data
train_data=pd.read_json('C:/Users/Yin/Desktop/train.json')
test_data=pd.read_json('C:/Users/Yin/Desktop/test.json')

#basic features
train_data["price_t"] =train_data["price"]/train_data["bedrooms"]#房间单价
test_data["price_t"] = test_data["price"]/test_data["bedrooms"] 
train_data["room_sum"] = train_data["bedrooms"]+train_data["bathrooms"] #浴室和房间总数
test_data["room_sum"] = test_data["bedrooms"]+test_data["bathrooms"] 
# count of photos #

train_data["num_photos"] = train_data["photos"].apply(len)#提供的照片数量
test_data["num_photos"] = test_data["photos"].apply(len)

# count of "features" #
train_data["num_features"] = train_data["features"].apply(len)#特征数量
test_data["num_features"] = test_data["features"].apply(len)
# count of words present in description column #
train_data["num_description_words"] = train_data["description"].apply(lambda x: len(x.split(" ")))#描述栏目的单词数量
test_data["num_description_words"] = test_data["description"].apply(lambda x: len(x.split(" ")))
#feature of created hour year month 
train_data["created"] = pd.to_datetime(train_data["created"])
test_data["created"] = pd.to_datetime(test_data["created"])
train_data["created_year"] = train_data["created"].dt.year
test_data["created_year"] = test_data["created"].dt.year
train_data["created_month"] = train_data["created"].dt.month
test_data["created_month"] = test_data["created"].dt.month
train_data["created_day"] = train_data["created"].dt.day
test_data["created_day"] = test_data["created"].dt.day
train_data["created_hour"] = train_data["created"].dt.hour
test_data["created_hour"] = test_data["created"].dt.hour

train_data["pos"] = train_data.longitude.round(3).astype(str) + '_' + train_data.latitude.round(3).astype(str)
test_data["pos"] = test_data.longitude.round(3).astype(str) + '_' + test_data.latitude.round(3).astype(str)

vals = train_data['pos'].value_counts()
dvals = vals.to_dict()
train_data["density"] = train_data['pos'].apply(lambda x: dvals.get(x, vals.min()))
test_data["density"] = test_data['pos'].apply(lambda x: dvals.get(x, vals.min()))
#可用特征名称
features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","price_t","price_per_room", "logprice", "density",
"num_photos", "num_features", "num_description_words","listing_id", "created_year", "created_month", "created_day", "created_hour"]


#定义运行xgboost构建预测模型
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'#data*class的矩阵 并且还有概率
    param['eta'] = 0.02#更新过程中用到的收缩步长，默认0.3，通常在0.01-0.2之间
    param['max_depth'] = 6#树的深度
    param['silent'] = 1#打印出运行时信息，为1时 沉默运行
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"#校验数据时所需要的评价指标
    param['min_child_weight'] = 1#孩子节点中最小权重和
    param['subsample'] = 0.7#用于训练模型的样本占样本总数的比例
    param['colsample_bytree'] = 0.7#在建立树时对特征随机采样的比例
    param['seed'] = seed_val#随机数种子
    num_rounds = num_rounds#boosting迭代计算次数

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
 
    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

#从manage_id 入手 十折交叉验证
#Step1:  将学习样本空间 C 分为大小相等的 K 份  
#Step2:  for i = 1 to K ：
 #           取第i份作为测试集
    #          for j = 1 to K:
   #               if i != j:
#                      将第j份加到训练集中，作为训练集的一部分
#                  end if
#              end for
#          end for
#  Step3:  for i in (K-1训练集)：
 #             训练第i个训练集，得到一个分类模型
 #             使用该模型在第N个数据集上测试，计算并保存模型评估指标
 #         end for
#  Step4:  计算模型的平均性能
#  Step5:  用这K个模型在最终验证集的分类准确率平均值作为此K-CV下分类器的性能指标

index=list(range(train_data.shape[0]))#train_data.shape([0]) train矩阵的第一梯度的长度 从0到train_data的第一梯度的一个list序列
random.shuffle(index)#对index进行随机排序
a=[np.nan]*len(train_data)#处理空值no.nan
b=[np.nan]*len(train_data)#处理空值
c=[np.nan]*len(train_data)#处理空值

for i in range(5):
    building_level={}
    for j in train_data['manager_id'].values:#manager_id 32为字符串
        building_level[j]=[0,0,0]#建立一个manage_id 对应的空矩阵 分别对应 low medium high 
    test_index=index[int((i*train_data.shape[0])/5):int(((i+1)*train_data.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_data.iloc[j]#使用位置选取数据
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_data.iloc[j]#使用位置选取数据
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_data['manager_level_low']=a
train_data['manager_level_medium']=b
train_data['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in train_data['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_data.shape[0]):
    temp=train_data.iloc[j]#使用位置选取数据
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_data['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
        
test_data['manager_level_low']=a
test_data['manager_level_medium']=b
test_data['manager_level_high']=c
#增加特征
features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')



categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_data[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_data[f].values) + list(test_data[f].values))
            train_data[f] = lbl.transform(list(train_data[f].values))
            test_data[f] = lbl.transform(list(test_data[f].values))
            features_to_use.append(f)

train_data['features'] = train_data["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_data['features'] = test_data["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
print(train_data["features"].head())
tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_data["features"])
te_sparse = tfidf.transform(test_data["features"])

train_X = sparse.hstack([train_data[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_data[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_data['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

#参考 magic_feature 评论中的代码 
image_date.loc[80240,"time_stamp"] = 1478129766 

image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
image_date["img_date_month"]            = image_date["img_date"].dt.month
image_date["img_date_week"]             = image_date["img_date"].dt.week
image_date["img_date_day"]              = image_date["img_date"].dt.day
image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
image_date["img_date_hour"]             = image_date["img_date"].dt.hour
image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

train_data = pd.merge(train_data, image_date, on="listing_id", how="left")
test_data = pd.merge(test_data, image_date, on="listing_id", how="left")



cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break
   
preds, model = runXGB(train_X, train_y, test_X, num_rounds=1000)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("data_submission.csv", index=False)
