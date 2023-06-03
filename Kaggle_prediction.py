#!/usr/bin/env python
# coding: utf-8

# # 1. 读取数据和划分训练集、测试集

# In[1]:


#读入在数据处理部分处理好的数据
import pandas as pd
data_full = pd.read_csv('data_withoutcrew.csv')


# In[7]:


data_full


# In[2]:


# 筛选出最终输入模型的特征列和目标列
data_full = data_full.drop(columns=['original_title','overview','poster_path','title','production_companies','production_countries'])


# In[3]:


data_full=data_full.drop(columns=['spoken_languages','status','tagline','keywords'])


# In[4]:


data_full=data_full.drop(columns=['Unnamed: 0'])


# In[5]:


data_full=data_full.drop(columns=['genres','release_date'])


# In[6]:


data_full=data_full.drop(columns=['cast','crew'])


# In[7]:


data_full=data_full.drop(columns=['original_language']) 


# In[8]:


data_full=data_full.drop(columns=['kr','western','Columbia Pictures Corporation','crew_Claude Chabrol','crew_Richard Thorpe','crew_Sidney Lumet','Warner Bros.','fr','crew_William A. Wellman','Toho Company','crew_Michael Curtiz','crew_Mervyn LeRoy','crew_Henry Hathaway','crew_Werner Herzog','crew_Raoul Walsh','crew_Fritz Lang','crew_Takashi Miike','crew_Robert Altman','es','crew_John Huston','crew_Ingmar Bergman','crew_Robert Wise','crew_John Ford','crew_Jean-Luc Godard','crew_Charlie Chaplin','crew_George Cukor',' Centre National de la Cinématographie (CNC)'])


# In[9]:


data_full.info()


# In[10]:


X = pd.DataFrame(data_full)
X.drop(['popularity','id'], axis = 1, inplace = True)
y = data_full.popularity


# In[6]:


features_all=['budget','genres','homepage','original_language','revenue','runtime','spoken_languages',
             'vote_average','vote_average','keywords','cast','crew','year','month',
             'action','adventure','animation','comedy','crime','documentary','drama','family','fantasy','fiction','foreign',
             'foreign','horror','movie','music','mystery','romance','science','thriller','tv','war','western','original_novel']


# In[ ]:


features=['vote_count','revenue','budget','homepage','vote_average','original_novel','year','runtime','original_language']


# In[6]:


features_4=['budget','revenue','vote_count','year','original_language','action', 'adventure', 'animation', 'comedy', 'crime',
       'documentary', 'drama', 'family', 'fantasy', 'fiction', 'foreign',
       'history', 'horror', 'movie', 'music', 'mystery', 'romance',
       'science', 'thriller', 'tv', 'war', 'western']


# In[62]:


features_5=['vote_count','revenue','budget','homepage','vote_average','original_novel','year','runtime','original_language','adventure','action','thriller']


# In[7]:


# 拆分出原来的训练数据集和测试数据集
train_X = data_full[:31801][features_all]
train_y = data_full[:31801].popularity
test_X = data_full[31801:][features_all]
train_X.shape, train_y.shape, test_X.shape


# In[11]:


train_X = X[:31801]
train_y = y[:31801]
test_X = X[31801:]
train_X.shape, train_y.shape, test_X.shape


# In[64]:


train_X.info()


# In[65]:


train_y.info()


# In[66]:


test_X.info()


# # 2. 模型构建

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


#导入需要的超参数选择器
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# ## 2.1 随机森林模型

# In[11]:


# 构建随机森林模型
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


# In[70]:


# 使用带交叉验证的网格搜索自动为随机森林模型搜索一个最佳决策树个数
from sklearn.model_selection import GridSearchCV 
parameters = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}
rfr_cv = GridSearchCV(estimator=rfr, param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
grid_result = rfr_cv.fit(train_X, np.ravel(train_y))
grid_result.best_params_


# In[19]:


# 预测测试数据
rfr_best = RandomForestRegressor(n_estimators=80,max_depth=31,max_features=9,bootstrap=True,n_jobs=-1)
rfr_best.fit(train_X, np.ravel(train_y))


# In[20]:


y_submit = rfr_best.predict(test_X)
y_submit


# ## 2.2 adaboost模型

# In[ ]:


# 使用带交叉验证的网格搜索训练一个最佳的AdaBoost模型，可以尝试调节参数：树的个数、学习率
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV 
parameters = {'n_estimators':[100,200,300,500,1000], 
              'learning_rate':[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
             }
ada = AdaBoostRegressor()
adacv = GridSearchCV(estimator=ada, param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
grid_result = adacv.fit(train_X, np.ravel(train_y))
grid_result.best_params_


# In[ ]:


# 预测测试数据
ada_best = AdaBoostRegressor(n_estimators=1250,learning_rate=0.3)
ada_best.fit(train_X, np.ravel(train_y))


# In[ ]:


y_submit = ada_best.predict(test_X)
y_submit


# ## 2.3 GDBT模型

# In[43]:


# 使用带交叉验证的网格搜索训练一个最佳的Gradient Boosting模型，可以尝试调节参数：树的个数、学习率、子采样、最大特征数
from sklearn.ensemble import GradientBoostingRegressor
parameters = {'n_estimators':[100,200,300,400,500,600], 
              'learning_rate':[0.05,0.1,0.15,0.2,0.3],
              'subsample':[0.5,0.6,0.7],
              'max_features':[None,'auto','log2']
             }
gbcv = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
grid_result = gbcv.fit(train_X, np.ravel(train_y))
grid_result.best_params_


# In[48]:


gb_best = GradientBoostingRegressor(n_estimators=400,
                                     learning_rate=0.15,
                                     subsample=0.5,
                                     max_features=None)
gb_best.fit(train_X_train, np.ravel(train_y_train))
y_pred=gb_best.predict(train_X_test)


# In[49]:


from sklearn import metrics
print("RMSE:",np.sqrt(metrics.mean_squared_error(train_y_test,y_pred)))


# In[50]:


y_submit = gb_best.predict(test_X)
y_submit


# ## 2.4 深度森林模型

# In[21]:


pip install deep-forest


# In[35]:


from deepforest import CascadeForestRegressor
dfr = CascadeForestRegressor(random_state=1)
dfr.fit(train_X, np.ravel(train_y))


# In[36]:


y_submit = dfr.predict(test_X)
y_submit = np.ravel(y_submit)


# ## 2.5 LightGBM模型

# In[14]:


pip install lightgbm


# In[30]:


import lightgbm as lgb
from lightgbm import plot_importance
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_depth': -1,
    'subsample': 0.8,
    'bagging_freq': 1,
    'feature_fraction ': 0.8,
    'learning_rate': 0.01,
    'bagging_fraction'=0.8,
    'lambda_l1'=0.5
}
dtrain = lgb.Dataset(train_X,train_y)
num_rounds = 180
model = lgb.train(params,dtrain, num_rounds, valid_sets=[dtrain], verbose_eval=10)


# In[12]:


# 使用带交叉验证的网格搜索自动为LightGBM搜索最佳参数
# 注：这个网格搜索是在学校机房的电脑上运行完成的，因此这个文件没有记录下运行结果
from sklearn.model_selection import GridSearchCV 
from lightgbm import LGBMRegressor

parameters = {'max_depth': [15, 20, 25, 30, 35],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'n_estimators':[100,200,300,400,500,600,1000]
}
gbm = LGBMRegressor(objective='regression',
                    booster='gbtree')
lgb_cv = GridSearchCV(estimator=gbm, param_grid=parameters, scoring='neg_mean_squared_error', verbose=1, cv=5, n_jobs=-1, return_train_score=True)
grid_result = lgb_cv.fit(train_X, np.ravel(train_y))
grid_result.best_params_


# In[31]:


y_submit = model.predict(test_X)
y_submit = np.ravel(y_submit)


# ## 2.6 CatBoost模型

# In[22]:


pip install catboost


# In[36]:


from catboost import CatBoostRegressor
cbr = CatBoostRegressor(iterations=350, 
                        learning_rate=0.01,
                        depth=10,
                        thread_count=-1)
cbr.fit(train_X, train_y)


# In[37]:


y_submit = cbr.predict(test_X)
y_submit = np.ravel(y_submit)


# ## 2.7 XGboost

# In[23]:


pip install xgboost


# In[10]:


from xgboost import XGBRegressor as XGBR
reg = XGBR(n_estimators=80,learning_rate=0.01).fit(train_X,train_y)


# In[13]:


y_submit = reg.predict(test_X)
y_submit = np.ravel(y_submit)


# ## 2.8 SVM

# In[13]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X= pd.DataFrame(min_max_scaler.fit_transform(X))
train_X = X[:31801]
train_y = y[:31801]
test_X = X[31801:]
train_X.shape, train_y.shape, test_X.shape


# In[19]:


from sklearn.svm import SVR
svmr = SVR(C=100,gamma=0.01,kernel='rbf')
svmr.fit(train_X,train_y)


# In[16]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [100, 1000], 'gamma': [0.01], 'kernel': ['linear','poly','rbf','sigmoid']} 
grid = GridSearchCV(SVR(), param_grid,scoring='neg_mean_squared_error' ,refit=True, verbose=1,n_jobs=-1,cv=5,return_train_score=True)
grid.fit(train_X,train_y)


# In[ ]:


y_submit = svmr.predict(test_X)
y_submit = np.ravel(y_submit)


# # 3. 模型聚合

# ## 3.1 Voting

# In[ ]:


# 注：这个模型是在学校机房的电脑上运行完成的，因此这个文件没有记录下运行结果
from sklearn.ensemble import VotingRegressor
estimators=[('df', CascadeForestRegressor(random_state=1)),
            ('lgb', LGBMRegressor(objective='regression', learning_rate=0.01, n_estimators=180)),
            ('rf', RandomForestRegressor(max_depth=31, max_features=9, n_estimators=80, n_jobs=-1))
           ]
vt = VotingRegressor(estimators=estimators)


# In[ ]:


vt.fit(train_X,train_y)


# In[ ]:


y_submit = st.predict(test_X)
y_submit = np.ravel(y_submit)


# ## 3.2 stacking

# In[39]:


from sklearn.ensemble import StackingRegressor
from deepforest import CascadeForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor as XGBR

estimators=[('df', CascadeForestRegressor(random_state=1)),
            ('lgb', LGBMRegressor(objective='regression', learning_rate=0.01, n_estimators=180)),
            ('rf', RandomForestRegressor(max_depth=31, max_features=9, n_estimators=80, n_jobs=-1))
           ]
st = StackingRegressor(estimators=estimators, final_estimator=ElasticNetCV())


# In[40]:


st.fit(train_X,train_y)


# In[ ]:


y_submit = st.predict(test_X)
y_submit = np.ravel(y_submit)


# # 4.预测结果的生成

# In[32]:


id = data_full[31801:].id
id


# In[37]:


submission = pd.DataFrame({'id': id , 'popularity': y_submit})
submission


# In[38]:


submission.to_csv('submission_30.csv', index=False)

