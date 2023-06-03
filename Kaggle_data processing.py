#!/usr/bin/env python
# coding: utf-8

# # 1. 数据处理
# ## 1.1 读取数据和概览 

# In[1]:


#导入pandas包和数据
import pandas as pd
data_train = pd.read_csv('movies_train.csv')
data_test=pd.read_csv('movies_test.csv')


# In[2]:


data_train.release_date[354]='2016-6-6'
data_train.release_date[601]='2003-10-7'
data_train.release_date[686]='2014-11-11'
data_train.release_date[1186]='2010-2-19'
data_train.release_date[1285]='1993-4-15'
data_train.release_date[2766]='2000-6-21'
data_train.release_date[2820]='2016-6-4'
data_train.release_date[3305]='2009-4-10'
data_train.release_date[4011]='2002-2-8'
data_train.release_date[5719]='2016-6-24'
data_train.release_date[6486]='2020-10-1'
data_train.release_date[6705]='2001-3-20'
data_train.release_date[7294]='1992-9-13'
data_train.release_date[7667]='1999-7-16'
data_train.release_date[7676]='2010-1-1'
data_train.release_date[9675]='2007-6-7'
data_train.release_date[10560]='2015-3-13'
data_train.release_date[11224]='2013-3-11'
data_train.release_date[11507]='1999-10-6'
data_train.release_date[12186]='2007-9-12'
data_train.release_date[12448]='1978-10-27'
data_train.release_date[12826]='2008-1-19'
data_train.release_date[13547]='2012-2-3'
data_train.release_date[14639]='2007-5-21'
data_train.release_date[16667]='2012-8-7'
data_train.release_date[17238]='1992-9-13'
data_train.release_date[17299]='2012-9-30'
data_train.release_date[18639]='2004-5-11'
data_train.release_date[18747]='2002-2-1'
data_train.release_date[19730]='1983-10-21'
data_train.release_date[19760]='2011-4-30'
data_train.release_date[19856]='1978-10-17'
data_train.release_date[20040]='2002-5-19'
data_train.release_date[21016]='2013-5-30'
data_train.release_date[22230]='2017-5-11'
data_train.release_date[22381]='2013-5-16'
data_train.release_date[23124]='2006-5-17'
data_train.release_date[23354]='1997-6-11'
data_train.release_date[23458]='1980-2-23'
data_train.release_date[23846]='2007-10-14'
data_train.release_date[23950]='2015-5-7'
data_train.release_date[24927]='2010-7-15'
data_train.release_date[25630]='1995-10-21'
data_train.release_date[25631]='2008-4-9'
data_train.release_date[26157]='2013-3-12'
data_train.release_date[27261]='2017-2-11'
data_train.release_date[27339]='2004-7-1'
data_train.release_date[27767]='2005-10-8'
data_train.release_date[28356]='2012-10-21'
data_train.release_date[29648]='1962-9-28'
data_train.release_date[30451]='2014-3-9'
data_train.release_date[30593]='2012-1-2'


# In[3]:


data_test.release_date[535]='2014-6-14'
data_test.release_date[2052]='1980-10-31'
data_test.release_date[2091]='2015-1-12'
data_test.release_date[2350]='2013-5-8'
data_test.release_date[2449]='2016-7-21'
data_test.release_date[3569]='2006-10-21'
data_test.release_date[4591]='2016-6-19'
data_test.release_date[4675]='2016-7-3'
data_test.release_date[5036]='2008-10-24'
data_test.release_date[5261]='2002-6-20'
data_test.release_date[5513]='2013-3-2'
data_test.release_date[6226]='2018-8-14'
data_test.release_date[6588]='1995-9-9'
data_test.release_date[6595]='2001-6-16'
data_test.release_date[6679]='2021-9-6'
data_test.release_date[6713]='1993-10-21'
data_test.release_date[6809]='1987-10-30'
data_test.release_date[7027]='2014-6-28'
data_test.release_date[7497]='1996-6-25'
data_test.release_date[7820]='2000-6-2'
data_test.release_date[8626]='2005-6-28'
data_test.release_date[8677]='2014-10-17'
data_test.release_date[8778]='2015-10-23'
data_test.release_date[535]='2014-6-14'
data_test.release_date[2052]='1980-10-31'
data_test.release_date[2091]='2015-1-12'
data_test.release_date[2350]='2013-5-8'
data_test.release_date[2449]='2016-7-21'
data_test.release_date[3569]='2006-10-21'
data_test.release_date[4591]='2016-6-19'
data_test.release_date[4675]='2016-7-3'
data_test.release_date[5036]='2008-10-24'
data_test.release_date[5261]='2002-6-20'
data_test.release_date[5513]='2013-3-2'
data_test.release_date[6226]='2018-8-14'
data_test.release_date[6588]='1995-9-9'
data_test.release_date[6595]='2001-6-16'
data_test.release_date[6679]='2021-9-6'
data_test.release_date[6713]='1993-10-21'
data_test.release_date[6809]='1987-10-30'
data_test.release_date[7027]='2014-6-28'
data_test.release_date[7497]='1996-6-25'
data_test.release_date[7820]='2000-6-2'
data_test.release_date[8626]='2005-6-28'
data_test.release_date[8677]='2014-10-17'
data_test.release_date[8778]='2015-10-23'
data_test.release_date[9274]='2001-12-6'
data_test.release_date[9511]='2012-9-28'
data_test.release_date[10216]='1979-12-1'
data_test.release_date[12515]='1993-9-16'
data_test.release_date[12610]='2013-7-23'
data_test.release_date[12920]='1937-10-20'
data_test.release_date[12944]='2019-7-21'
data_test.release_date[8808]='2012-9-18'
data_test.release_date[10710]='2008-10-12'


# In[4]:


data_test.original_language[1932]='fr'
data_test.original_language[4756]='en'
data_test.original_language[7698]='en'
data_test.original_language[11301]='en'


# In[5]:


data_train.original_language[2022]='fr'
data_train.original_language[4552]='ja'
data_train.original_language[5456]='en'
data_train.original_language[6196]='en'
data_train.original_language[15955]='en'
data_train.original_language[16612]='fr'
data_train.original_language[19339]='en'


# In[6]:


#合并训练数据和测试数据以便于后续处理
data_full=data_train.append(data_test, ignore_index=True)


# In[7]:


#查看数据前5行
data_full.head()


# In[8]:


#查看数据后五行
data_full.tail()


# In[9]:


#显示行数和列数
print('训练数据集:',data_train.shape)
print('测试数据集:',data_test.shape)
print('全部数据集:',data_full.shape)


# In[10]:


#显示所有列的数据类型和非空值个数
data_full.info()


# 可以看到23列数据中有7列是数值型特征（其中id和budget列是整数类型，popularity、revenue、runtime、vote_average和vote_count列是小数类型），16列是非数值型特征。一些列，如中存在homepage、original_language，非空值的个数小于45430，说明这些列中存在缺失值。我们后续进行数据预处理时再讨论如何处理缺失数据.
# （值得处理的缺失数据列：original_language【用production_companies production_countries spoken_languages预测】、runtime、release_date）

# ## 1.2 数值型数据的处理

# In[11]:


#显示数值型特征列的描述统计信息
data_full.describe()


# In[12]:


data_full.revenue.describe()


# ### 1.2.1 release_date的处理

# In[13]:


date=data_full['release_date']
date


# In[14]:


#将日期分割并转化为数值型数据
date=data_full['release_date']
date=pd.to_datetime(date)
date


# In[15]:


date[1].year
date[1].month


# In[16]:


def get_ymd(date):
    # 这里的输入date是一列年月日数据
    Y,M =[],[]
    for i in range(len(date)):
        oneday=date[i]
        year=oneday.year
        month=oneday.month

        Y.append(year)
        M.append(month)
        
    date=pd.DataFrame()
    date['year']=Y
    date['month']=M
    return date


# In[17]:


date = get_ymd(date)
date


# In[18]:


date.info()


# In[19]:


data_full = pd.concat([data_full, date], axis=1)
data_full


# ### 1.2.3 缺失值的处理

# In[20]:


data_full.isnull().sum()


# In[21]:


len(data_train[data_train.revenue==0])


# In[22]:


len(data_test[data_test.revenue==0])


# In[23]:


len(data_full[data_full.revenue==0])


# In[24]:


len(data_full[data_full.budget==0])


# In[25]:


len(data_train[data_train.budget==0])


# In[26]:


len(data_test[data_test.budget==0])


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(data_train.corr(), annot=True, vmax=1, square=True, cmap='Blues')


# 由相关系数图可以看出对于revenue而言，与其最相关的是popularity, budget和vote_count，由于popularity是待预测的序列，我们使用budget和vote_count预测revenue

# In[28]:


data_full.revenue.describe()


# In[29]:


data_full.revenue.sort_values()


# In[30]:


data_full.vote_count.sort_values()


# In[31]:


#处理revenue的缺失值
import numpy as np
feature_cols_re = ['vote_count','budget']
y_re = data_full[(data_full.revenue!=0) & (data_full.budget!=0)].revenue
X_re = data_full[(data_full.revenue!=0 )& (data_full.budget!=0)][feature_cols_re]
X_pre = data_full[data_full.revenue==0][feature_cols_re]


# In[32]:


y_re = data_full[(data_full.revenue!=0) & (data_full.budget!=0)].revenue
y_re


# In[33]:


X_re = data_full[(data_full.revenue!=0 )& (data_full.budget!=0)][feature_cols_re]
X_re


# In[34]:


X_pre = data_full[data_full.revenue==0][feature_cols_re]
X_pre


# In[35]:


from sklearn.ensemble import RandomForestRegressor
model_re = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
model_re.fit(X_re, y_re)
re = model_re.predict(X_pre)


# In[36]:


pd.DataFrame(re)


# In[37]:


import warnings
warnings.filterwarnings("ignore")
ls_re = data_full[data_full.revenue==0].index.tolist()
for i in ls_re:
    data_full.revenue[i] = re[ls_re.index(i)]


# In[38]:


data_full.revenue.describe()


# In[39]:


data_full.budget.describe()


# In[40]:


#同理，处理budget的缺失值
feature_cols_bg = ['vote_count','revenue']
y_bg = data_full[(data_full.revenue!=0) & (data_full.budget!=0)].budget
X_bg = data_full[(data_full.revenue!=0 )& (data_full.budget!=0)][feature_cols_bg]
X_pbg = data_full[data_full.budget==0][feature_cols_bg]


# In[41]:


model_bg = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
model_bg.fit(X_bg, y_bg)
bg = model_re.predict(X_pbg)


# In[42]:


import warnings
warnings.filterwarnings("ignore")
ls_bg = data_full[data_full.budget==0].index.tolist()
for i in ls_bg:
    data_full.budget[i] = bg[ls_bg.index(i)]


# In[43]:


data_full.budget.describe()


# In[44]:


data_full.runtime.isnull().sum()


# In[45]:


x=data_full.runtime
y=data_full.popularity                                                                       
plt.scatter(x,y)


# In[46]:


len(data_full[data_full.runtime==0])


# 从上图数据表示，受欢迎程度与电影时长的关系符合正态分布。太长或太短都影响受欢迎程度。

# In[47]:


data_full.runtime.mean()


# In[48]:


data_full.runtime.median()


# In[49]:


#用平均值补runtime的缺失值和0值
#填补缺失值
data_full.runtime.fillna(data_full.runtime.mean(), inplace=True)
#填补0值
data_full[data_full.runtime==0].runtime=data_full.runtime.mean()


# In[50]:


data_full.isnull().sum()


# 核实完毕，需要的特征的缺失值都处理完了

# ## 1.3 文本型数据处理
# ### 1.3.1 genres的处理

# In[51]:


#将genres中的类型转成独热编码
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(data_full['genres'])
vect.get_feature_names_out()


# In[52]:


genres_dtm = vect.transform(data_full['genres'])
genres_dtm


# In[53]:


genres_dtm.toarray()


# In[54]:


genreclass=pd.DataFrame(genres_dtm.toarray(), columns=vect.get_feature_names_out())
genreclass


# In[55]:


genreclass.sum().sort_values(ascending=False)


# In[56]:


#将新生成的独热编码拼接到原来的数据集中
data_full = pd.concat([data_full, genreclass], axis=1)
data_full


# ### 1.3.1 production_countries的处理

# In[57]:


#将productiion_countries中的类型转成独热编码
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(data_full['production_countries'])
vect.get_feature_names_out()


# In[58]:


countries_dtm = vect.transform(data_full['production_countries'])
countries_dtm


# In[59]:


countries_dtm.toarray()


# In[60]:


countryclass=pd.DataFrame(countries_dtm.toarray(), columns=vect.get_feature_names_out())
countryclass


# In[61]:


countryclass.sum().sort_values(ascending=False).head(20)


# In[62]:


countries_col=['us','gb','fr','de','it','ca','jp','es','ru','in','hk','se','au','kr']
country_class=countryclass[countries_col]
country_class


# In[63]:


data_full = pd.concat([data_full, country_class], axis=1)
data_full


# ### 1.3.1 production_companies的处理

# In[64]:


data_full['production_companies']=data_full['production_companies'].str.strip('[]').str.replace("","").str.replace("'","")
data_full['production_companies']=data_full['production_companies'].str.split(',')

list1=[]
for i in data_full['production_companies']:
    list1.extend(i)
    
def sort_count(list):
    list2=[]
    result=[]
    for i in set(list):
        list2.append([list.count(i),i])
    
    list2.sort(reverse=True)
    
    for count,num in list2:
        for i in range(count):
            result.append(num)
    return result
list3=sort_count(list1)
list_production_companies=[]

for i in list3:
    if i not in list_production_companies:
        list_production_companies.append(i)
list_production_companies.remove("")
print(len(list_production_companies))
list_production_companies=list_production_companies[0:25]


# In[65]:


list_production_companies


# In[66]:


data_full['production_companies']


# In[67]:


#独热编码production
for per in list_production_companies:
    data_full[per]=0
    
    z=0
    for gen in data_full['production_companies']:
        
        if per in list(gen):
            data_full.loc[z,per]=1
        else:
            data_full.loc[z,per]=0
        z+=1


# In[68]:


data_full


# ### 1.3.2 original_language的处理

# In[69]:


#将original_language分类
data_full.original_language.value_counts().head(20)


# In[70]:


data_full['EN']=data_full['original_language'].map(lambda x: 1 if str(x)=='en' else 0)
data_full['FR']=data_full['original_language'].map(lambda x: 1 if str(x)=='fr' else 0)
data_full['IT']=data_full['original_language'].map(lambda x: 1 if str(x)=='it' else 0)
data_full['JA']=data_full['original_language'].map(lambda x: 1 if str(x)=='ja' else 0)
data_full['DE']=data_full['original_language'].map(lambda x: 1 if str(x)=='de' else 0)
data_full['ES']=data_full['original_language'].map(lambda x: 1 if str(x)=='es' else 0)
data_full['RU']=data_full['original_language'].map(lambda x: 1 if str(x)=='ru' else 0)


# In[71]:


data_full


# In[72]:


#将original_language分类，分成英文非英文
data_full.original_language.value_counts()


# In[74]:


#以下创建二值变量
data_full['original_language'] = data_full ['original_language'].map(lambda s: 1 if s=='en' else 0)
data_full['original_language']
data_full.original_language.value_counts()


# ### 1.3.3 crew和cast的处理

# In[47]:


#提取crew中的导演名
#for i in range(0,45430):
    #d1=eval(data_full.crew[i])
    #for x in d1:
        #if x['job']=='Director':
            #a=x['name']
            #data_full.crew[i]=a


# In[76]:


#data_full.crew.value_counts().head(25)


# In[46]:


#popularity_of_director = data_train.groupby('crew').popularity.mean()
#popularity_of_director


# In[123]:


#crew_mapDict={"Clint Eastwood":"Clint Eastwood","Martin Scorsese":"Martin Scorsese","Woody Allen":"Woody Allen","Alfred Hitchcock":"Alfred Hitchcock",}


# In[124]:


#data_full['crew']=data_full['crew'].map(crew_mapDict)


# In[125]:


#director=pd.get_dummies(data_full['crew'],prefix='crew',drop_first=False)


# In[126]:


#director


# In[127]:


#data_full=pd.concat([data_full,director],axis=1)


# ### 1.3.4 homepage的处理

# In[73]:


data_full.homepage.describe()


# In[74]:


data_full['homepage'] = data_full ['homepage'].map(lambda p: 0 if str(p)=='nan' else 1)
data_full['homepage']
data_full.homepage.value_counts()


# In[75]:


data_train['homepage'] = data_train ['homepage'].map(lambda p: 0 if str(p)=='nan' else 1)
data_train['homepage']
data_train.homepage.value_counts()


# In[76]:


popularity_of_hp = data_train.groupby('homepage').popularity.mean()
popularity_of_hp.sort_values()


# In[77]:


popularity_of_hp = data_train.groupby('homepage').popularity.mean()
x=[0,1]
y=[2.48505,5.058810]
plt.bar(x,y)


# ### 1.3.5 keywords的处理

# In[78]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(data_full['keywords'])
vect.get_feature_names_out()


# In[79]:


#原创电影与改编电影对比分析
original_novel = pd.DataFrame()
original_novel['keywords'] = data_train['keywords'].str.contains('based on').map(lambda x: 1 if x else 0)
original_novel['popularity'] = data_train['popularity']
original_novel.keywords.value_counts()


# In[80]:


popularity_of_original = original_novel.groupby('keywords').popularity.mean()
popularity_of_original.sort_values()


# In[81]:


x=[0,1]
y=[2.758892,7.274313]
plt.bar(x,y)


# 可以看出改编的电影的平均热度要高于非改编的电影

# In[82]:


#在完整数据集中加入是否改编的特征列
data_full['original_novel']=data_full['keywords'].str.contains('based on').map(lambda x: 1 if x else 0)


# In[83]:


# 查看每个特征与popularity的相关系数，并按绝对值的降序排列
corrDf = data_full.corr()
corrDf['popularity'].map(abs).sort_values(ascending =False)


# In[84]:


data_full.to_csv(r'C:/Users/lenovo/Desktop/data_submitted.csv')


# In[ ]:




