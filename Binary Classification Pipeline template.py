#!/usr/bin/env python
# coding: utf-8

# 
# # Content <a name="toc"></a>
# 1. [Upload Train Dataset](#p1)
# 2. [Modelling](#p2)
# 3. [Validation & Feature Selection](#p3)
# 4. [HyperOpt Optimization](#p4)
# 5. [PSI Calculation](#p5)
# 6. [Model report](#p6)

# ## Import necessary libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from scipy.stats import ks_2samp
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import matplotlib.pyplot as plt


# In[4]:


import xgboost as xgb
import hyperopt
import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import sys
import pickle
import dill


# In[5]:


import math
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve


# 
# # 1. Upload Train dataset <a name="p1"></a>

# In[6]:


start = datetime.now()
print(start)
data = pd.read_csv('VSP_visits_forecast_sample200k.csv', engine='c')
stop = datetime.now()
print(stop)
print('Upload Time: ',stop-start)


# In[7]:


data.info()


# In[5]:


# Drop all targets except chosen one
data = data.drop(columns=['cnt_visit_1Q', 'min_card_exp_dt'])


# In[8]:


# Select all columns with 'dt' in names
data.filter(regex='dt').head(10)


# In[6]:


x = data.drop(['target_vsp', 'id_target'], axis=1)
y = data.target_vsp


# In[7]:


y.value_counts()


# In[10]:


plt.figure(figsize=(7,5))
plt.grid()
sns.countplot(y)
plt.xticks((0,1), ['Не пришёл в ВСП: {:.2%}'.format(y.value_counts()[0]/y.value_counts().sum()),
                   'Пришёл в ВСП: {:.2%}'.format(y.value_counts()[1]/y.value_counts().sum())])
plt.ylabel('Количество объектов')
plt.show()


# In[11]:


x.dtypes.value_counts()


# In[8]:


obj_var = list(x.dtypes[x.dtypes==object].index)


# # 2.Modelling <a name="p2"></a>

# ## Encoder for categorical decoding

# In[9]:


class MyEncoder:
    
    def __init__(self, name):
        self.name = name
        self.columns = None
        self.dict = {}
    
    def fit(self, X, list_to_encode):
        self.columns = list_to_encode
        for column in list_to_encode:
            self.dict[column] = LabelEncoder().fit(X[column].astype(str))
            
    def transform(self, X):
        for column in self.columns:
            X[column] = self.dict[column].transform(X[column].astype(str))
            
    def fit_transform(self, X, list_to_encode: list):
        self.fit(X, list_to_encode)
        self.transform(X)


# In[10]:


pe = MyEncoder('try1')


# In[11]:


start = datetime.now()
print(start)
pe.fit_transform(x, obj_var)
stop = datetime.now()
print(stop)
print('Upload Time: ',stop-start)


# In[16]:


# Save our trained Encoder
pkl_filename = "MyEncodeTrain.dill"
with open (pkl_filename, 'wb') as file:
    dill.dump(pe,file)


# In[36]:


list_of_index_good_nan = x.isna().sum().sort_values().head(1500).index.to_list()


# In[37]:


obj_var = x[list_of_index_good_nan].dtypes[x[list_of_index_good_nan].dtypes==object].index.to_list()


# In[39]:


len(obj_var)


# In[38]:


num_var = x[list_of_index_good_nan].drop(x[list_of_index_good_nan].dtypes[x[list_of_index_good_nan].dtypes==object].index.to_list(), axis=1).columns.to_list()


# In[20]:


num_var


# In[15]:


# Проверка влияния факторов на таргет
var_dict = []

for i in x[obj_var+num_var]:
    table = pd.DataFrame({'y': y.values, str(i): x[i].values})
    table.dropna(inplace=True)
    table.sort_values(by=str(i), ascending=0, inplace=True)
    table[str(i)] = minmax_scale(table[str(i)])
    var_dict.append(roc_auc_score(table['y'], table[str(i)]))
#    print(i)


# In[16]:


result_table = pd.DataFrame({'Variable': obj_var+num_var, 
                             'AUC ROC': var_dict}).sort_values(by='AUC ROC', ascending=0)


# In[17]:


result_table['GINI'] = result_table['AUC ROC'].apply(lambda x: -1+2*x if x>= 0.5 else -1+2*(1-x))


# In[18]:


result_table = result_table.sort_values(by='GINI', ascending=False)


# In[19]:


result_table[result_table['GINI']>=0.1]


# ## Select top features with Gini >= 0.05

# In[20]:


important_var = result_table[result_table['GINI']>=0.05]['Variable'].to_list()


# In[24]:


important_var


# In[21]:


# Число значимых фичей с Gini >= 0.05
len(important_var)


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x[important_var], y, test_size = 0.3, random_state=0)


# In[23]:


# Hyper parameters here are defined by HyperOpt selection (see below)
clf = CatBoostClassifier(iterations=1000, learning_rate=0.24494855927037054, depth=8,
                         l2_leaf_reg = 9.629585619079913 , scale_pos_weight=8, early_stopping_rounds = 50)


# In[24]:


clf.fit(x_train, y_train, eval_set=(x_test,y_test), verbose=False,plot=True)


# In[25]:


print('Проверка на качество (для всех Variables)')
print('')
print('Gini Train: {0}'.format(-1+2*roc_auc_score(y_train, clf.predict_proba(x_train)[:,1]))) 
print('GINI Validation: {0}'.format(-1+2*roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])))


# In[33]:


# Feature Importance by Catboost


# In[26]:


FI = pd.DataFrame({'Variable': x_test.columns.tolist(), 'FI': clf.feature_importances_}).sort_values(by='FI', ascending=0)


# In[27]:


var_100 = FI.sort_values('FI', ascending=False)[:100]['Variable'].tolist()


# In[28]:


FI.sort_values('FI', ascending=False)[:20]


# # 3.Validation & Feature selection <a name="p3"></a>

# # График для Validation

# In[29]:


N = 20
gini_score_cat = []
for i in range(0, N):
    var_N = FI['Variable'].to_list()[:i+1]
    clf.fit(x_train[var_N], y_train, verbose=0)
    print(str(i+1)+' -> ROC AUC: {0} _//////_ GINI: {1}'.format(
        round(roc_auc_score(y_test,
                            clf.predict_proba(x_test[var_N])[:,1]),4),
        round(-1+2*roc_auc_score(y_test,
                            clf.predict_proba(x_test[var_N])[:,1]),4)))
    print(FI['Variable'].to_list()[i])
    print('')
    gini_score_cat.append(-1+2*roc_auc_score(y_test,
                            clf.predict_proba(x_test[var_N])[:,1]))


# In[30]:


plt.plot(list(range(len(gini_score_cat))), gini_score_cat)
plt.axis([0,21,0.1,.8])
plt.xlabel('N factors')
plt.ylabel('GINI score')
plt.show()


# ## Select top 11 features with max Gini value

# In[31]:


top_11 = FI['Variable'].to_list()[:11]


# In[32]:


top_11


# In[33]:


clf_top = CatBoostClassifier(iterations=1000, max_depth=8,learning_rate = 0.24494855927037054,
                             scale_pos_weight=8,l2_leaf_reg = 9.629585619079913 ,early_stopping_rounds=50)


# In[34]:


clf_top .fit(x_train[top_11], y_train, eval_set=(x_test[top_11],y_test), verbose=False,plot=True)


# In[35]:


print('Проверка на лучших 11 факторах качество (для всех Variables)')
print('')
print('Gini Train: {0}'.format(-1+2*roc_auc_score(y_train, clf_top.predict_proba(x_train[top_11])[:,1]))) 
print('GINI Validation: {0}'.format(-1+2*roc_auc_score(y_test, clf_top.predict_proba(x_test[top_11])[:,1])))


# In[36]:


# Save our trained model 
pkl_filename = "catboost_TOP_var.pkl"
with open (pkl_filename, 'wb') as file:
    pickle.dump(clf_top,file)


# In[ ]:





# # OOT 

# In[12]:


OOT_test = pd.read_csv('VSP_visits_data_OOT_new1.csv', engine ='c')


# In[13]:


OOT_test.shape


# In[39]:


# Upload model from pickle file
pkl_filename = "catboost_TOP_var.pkl"
with open(pkl_filename, 'rb') as file:
    clf_top = pickle.load(file)


# In[13]:


# Drop all targets except chosen one
OOT_test = OOT_test.drop(columns=['cnt_visit_1Q', 'min_card_exp_dt'])


# In[14]:


x_test_oot = OOT_test.drop(['target_vsp', 'id_target'], axis=1)
y_test_oot = OOT_test.target_vsp


# In[15]:


# Select all "object" type columns
obj_var_encode = list(x_test_oot.dtypes[x_test_oot.dtypes==object].index)


# In[16]:


# Upload Class from pickle file
pkl_filename = "MyEncodeTrain.dill"
with open(pkl_filename, 'rb') as file:
    pe_predict = dill.load(file)


# In[17]:


pe_predict.fit(x_test_oot, obj_var_encode)
pe_predict.transform(x_test_oot)


# In[45]:


print('Проверка на качество (для OOT)')
print('')
print('GINI OOT  Test: {0}'.format(-1+2*roc_auc_score(y_test_oot, clf_top.predict_proba(x_test_oot)[:,1])))


# # 4.HyperOpt optimization <a name="p4"></a>

# ## Searching for best hyper parameters

# In[ ]:


from hyperopt import hp, tpe, STATUS_OK, Trials


# In[ ]:


# Upload sample of our dataset to minimize optimization time
data_sample = pd.read_csv('VSP_visits_forecast_sample200k.csv')


# In[ ]:


data_sample.head(10)
data_sample.shape


# In[ ]:


x_sample = data_sample.drop(['target_vsp', 'id_target'], axis=1)
y_sample = data_sample.target_vsp


# In[ ]:


# Select all "object" type columns
obj_var = list(x_sample.dtypes[x_sample.dtypes==object].index)


# In[ ]:


# Make Encoding to all "object" type columns
pe = PredictEncode('try1')
start = datetime.now()
print(start)
pe.fit_transform(x_sample, obj_var)
stop = datetime.now()
print(stop)
print('Upload Time: ', stop-start)


# ## Initialize functions for searching optimal hyper parameters

# In[1]:


def objective(space):
    clf_opt = CatBoostClassifier(iterations = space['n_estimators'],
                                 learning_rate = space['learning_rate'],
                                 depth = space['depth'],
                                 l2_leaf_reg = space['l2_leaf_reg']
                                )


# In[2]:


class UciAdultClassifierObjective(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0
        
    def _to_catboost_params(self, hyper_params):
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg']}
    
    # hyperopt optimizes an objective using `__call__` method (e.g. by doing 
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters 
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)
        
        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()
        
        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=4242,
            verbose=False)
        
        # scores returns a dictionary with mean and std (per-fold) of metric 
        # value for each cv iteration, we choose minimal value of objective 
        # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        max_mean_auc = np.max(scores['test-AUC-mean']) # change 'max' on 'avg'
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)
        
        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)
        
        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}


# In[3]:


def find_best_hyper_params(dataset, const_params, max_evals=100):    
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 0.5),
        'depth': hyperopt.hp.randint('depth',10),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 20)}
    objective = UciAdultClassifierObjective(dataset=dataset, const_params=const_params, fold_count=2)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=4242))
    return best

def train_best_model(X, y, const_params, max_evals=100, use_default=False):
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each 
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes != np.float)[0])
    
    if use_default:
        # pretrained optimal parameters
        best = {
            'learning_rate': 0.1, 
            'depth': 5, 
            'l2_leaf_reg': 1}
    else:
        best = find_best_hyper_params(dataset, const_params, max_evals=max_evals)
    
    # merge subset of hyper-parameters provided by hyperopt with hyper-parameters 
    # provided by the user
    hyper_params = best.copy()
    hyper_params.update(const_params)
    
    # drop `use_best_model` because we are going to use entire dataset for 
    # training of the final model
    hyper_params.pop('use_best_model', None)
    
    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)
    
    return model, hyper_params


# In[ ]:


# make it True if your want to use GPU for training
have_gpu = False
# skip hyper-parameter optimization and just use provided optimal parameters
use_optimal_pretrained_params = False
# number of iterations of hyper-parameter search
hyperopt_iterations = 20

const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC', 
    'custom_metric': ['AUC'],
    'iterations': 100,
    'random_seed': 4242})

model, params = train_best_model(
    x_sample, y_sample, 
    const_params, 
    max_evals=hyperopt_iterations, 
    use_default=use_optimal_pretrained_params)
print('best params are {}'.format(params), file=sys.stdout)


# In[ ]:





# # 5.PSI calculation by selected features <a name="p5"></a>

# In[50]:


# Functions: 
# 1. usr_al_psi - PSI between two lists od numbers
# 2. psi_df - PSI between all columns of two dataframes. Uses usr_al_psi func. Also calculates PSI for categorical variables
# 3. action_with_warnings - Supress WARNINGs in the log

# Functions: 
# 1. usr_al_psi - PSI between two lists od numbers
# 2. psi_df - PSI between all columns of two dataframes. Uses usr_al_psi func. Also calculates PSI for categorical variables
# 3. action_with_warnings - Supress WARNINGs in the log

def usr_al_psi(dts1, dts2, n_groups):
    dts1_prc = []
    dts2_prc = []
    psi = []
    null_cat = 'N'
    if len(dts1[~dts1.notnull()]) * len(dts2[~dts2.notnull()]) > 0:
        psi_null = (len(dts1[~dts1.notnull()]) / len(dts1) - len(dts2[~dts2.notnull()]) / len(dts2)) *             math.log((len(dts1[~dts1.notnull()]) / len(dts1))/(len(dts2[~dts2.notnull()]) / len(dts2)))
    elif len(dts1[~dts1.notnull()]) + len(dts1[~dts1.notnull()]) == 0:
        psi_null = 0
    else:
        psi_null = 1
        if len(dts1[~dts1.notnull()]) > 0:
            null_cat = 'T'
        else:
            null_cat = 'V'
    dts1 = dts1[dts1.notnull()]
    dts2 = dts2[dts2.notnull()]
    dts1 = dts1.tolist()
    dts2 = dts2.tolist()
    dts1.sort()
    psi_group = set(dts1[int(i * len(dts1) / n_groups)] for i in range(1, n_groups))
    psi_group = list(psi_group)
    psi_group.sort()
    psi_txt_group=[]
    j = 0
    while j < len(psi_group):
        if j == 0:
            dts1_prc.append(len([dts1[i] for i in range(0, len(dts1)) if dts1[i] <= psi_group[j]])/len(dts1))
            dts2_prc.append(len([dts2[i] for i in range(0, len(dts2)) if dts2[i] <= psi_group[j]])/len(dts2))
            psi_txt_group.append('X <= ' + str(psi_group[j]))
        else:
            dts1_prc.append(len([dts1[i] for i in range(0, len(dts1)) if dts1[i] <= psi_group[j] and dts1[i] > psi_group[j - 1]])/len(dts1))
            dts2_prc.append(len([dts2[i] for i in range(0, len(dts2)) if dts2[i] <= psi_group[j] and dts2[i] > psi_group[j - 1]])/len(dts2))
            psi_txt_group.append(str(psi_group[j - 1]) + ' < X <= ' + str(psi_group[j]))
        if j == len(psi_group) - 1:
            dts1_prc.append(len([dts1[i] for i in range(0, len(dts1)) if dts1[i] > psi_group[j]])/len(dts1))
            dts2_prc.append(len([dts2[i] for i in range(0, len(dts2)) if dts2[i] > psi_group[j]])/len(dts2))
            psi_txt_group.append('X > ' + str(psi_group[j]))
        j += 1
    psi_df = pd.DataFrame(psi_txt_group, columns=['group']).join(pd.DataFrame(dts1_prc, columns=['dts1_prc'])).join(pd.DataFrame(dts2_prc, columns=['dts2_prc']))
    psi_df['psi'] = 0
    psi_df['psi'] [psi_df.dts1_prc != psi_df.dts2_prc] = (psi_df.dts1_prc - psi_df.dts2_prc) * ((psi_df.dts1_prc+0.00001) / psi_df.dts2_prc).apply(math.log)
    if psi_null != 1:
        return sum(psi_df.psi) + psi_null, ''
    else:
        return "%.8f" % sum(psi_df.psi), 'Non-matching NaN is found. (' + null_cat + ') PSI -> without NaN'

def psi_df(df1, df2, path):
    psi_cat = []
    psi_df = pd.DataFrame(columns=['Variable','PSI', 'Comment'])
    j = 0
    for i in df1.columns.tolist():
        if ('int' in str(df1[df1[i].notnull()][i].dtype) or 'float' in str(df1[df1[i].notnull()][i].dtype)) and (df1[i].unique().shape[0] > 3 or (df1[i].unique().shape[0] > 2 and not (df1[i].isnull().values.any()))):
            if type(usr_al_psi(df1[i], df2[i].astype(float), 10)) != str:
                psi_df.loc[j] = [i, "%.8f" % float(usr_al_psi(df1[i], df2[i].astype(float), 10)[0]), usr_al_psi(df1[i], df2[i].astype(float), 10)[1]]
#                 print(i, ': ', "%.8f" % usr_al_psi(df1[i], df2[i].astype(float), 10))
            else:
                psi_df.loc[j] = [i, float(usr_al_psi(df1[i], df2[i].astype(float), 10)[0]), usr_al_psi(df1[i], df2[i].astype(float), 10)[1]]
#                 print(i, ': ', usr_al_psi(df1[i], df2[i].astype(float), 10))
        else:
            df1[i] = df1[i].astype(str)
            df2[i] = df2[i].astype(str)
            df1[i].fillna('MISSING', inplace=True)
            df2[i].fillna('MISSING', inplace=True)
            if sorted(df1[i].unique().tolist()) == sorted(df2[i].unique().tolist()):
                for cat in df1[i].unique():
                    ben_pct = (df1[df1[i]==cat][i].count() +0.0) / df1.shape[0]
                    comp_pct = (df2[df2[i]==cat][i].count() +0.0) / df2.shape[0]
                    psi_cat.append((ben_pct-comp_pct)*math.log(ben_pct/comp_pct))                
                psi=sum(psi_cat)
                psi_df.loc[j] = [i, "%.8f" % psi, '']
#                 print(i, ': ', "%.8f" % psi)
            else:
                psi=sum(psi_cat)
                psi_df.loc[j] = [i, "%.8f" % psi, 'Non-matching categories are found: ' +                       str([set(df1[i].unique()) - set(df2[i].unique()), set(df2[i].unique()) - set(df1[i].unique())]) +                         ' PSI -> only common categories']
#                 print(i, ': ', 'Non-mathcing categories are found: ', end = '')
#                 print([set(df1[i].unique()) - set(df2[i].unique()), set(df2[i].unique()) - set(df1[i].unique())], end = ' ')
#                 print('PSI(only common categories): ', "%.8f" % psi)
        j += 1
    if path != '':
        psi_df.to_csv(path, index=None)
    return psi_df

def action_with_warnings():
    warnings.warn("should not appear")


# In[51]:


for i in top_11:
    print(i)
    print(usr_al_psi(x_train[i], x_test_oot[i], 10))
    print()


# In[53]:


top_9 = FI['Variable'].to_list()[:11]
top_9.remove('prd_client_activity_1m_ind')
top_9.remove('lbt_acct_dep_td_bal_rub_amt')


# In[18]:


top_9 = ['prd_mnth_expiration_card_qty',
 'prd_mnth_expiration_deposit_qty',
 'avg_fist_V_1y',
 'crd_dc_active_open_qty',
 'crd_dc_active_open_qty_3AVG',
 'lbt_acct_tot_davg_mnth_rub_amt_3',
 'lbt_inf_total_qty_3AVG',
 'max_cnt_1y',
 'crd_dc_act_all_qty_3min']


# In[79]:


top_9


# In[68]:


#### what if top_10


# In[23]:


clf_9 = CatBoostClassifier(iterations=1000, max_depth=8,learning_rate = 0.24494855927037054,
                             scale_pos_weight=8,l2_leaf_reg = 9.629585619079913 ,early_stopping_rounds=50)


# In[24]:


clf_9.fit(x_train[top_9], y_train)


# In[25]:


prediction_oot_test = clf_9.predict_proba(x_test_oot[top_9])[:,1]


# In[26]:


predictio_oot = pd.Series(prediction_oot_test)


# In[28]:


df_helper = pd.DataFrame({'y_score': clf_9.predict_proba(x_test_oot[top_9])[:,1],
                      'y_true': y_test_oot}) 


# In[9]:


## False attempts to make scores calibration


# In[46]:


fop, mpv = calibration_curve(df_helper['y_true'], df_helper['y_score'])


# In[47]:


plt.plot([0, 1], [0, 1], linestyle ='--')
plt.plot(mpv,fop, marker='.')
plt.show()


# In[66]:


mpv


# In[67]:


fop


# In[57]:


cal = CalibratedClassifierCV(clf_9, cv='prefit', method='isotonic')


# In[58]:


cal.fit(x_test[top_9],y_test)


# In[59]:


yhat = cal.predict(x_test_oot[top_9])


# In[60]:


df_helper = pd.DataFrame({'y_score': cal.predict_proba(x_test_oot[top_9])[:,1],
                      'y_true': y_test_oot}) 


# In[ ]:





# In[57]:


print('Проверка на лучших 9 факторах на OOT')
print('')
print('GINI: {0}'.format(-1+2*roc_auc_score(y_test_oot, clf_9.predict_proba(x_test_oot[top_9])[:,1])))
print('ROC AUC: {0}'.format(roc_auc_score(y_test_oot, clf_9.predict_proba(x_test_oot[top_9])[:,1])))
print('Precision: {0}'.format(precision_score(y_test_oot, clf_9.predict(x_test_oot[top_9]))))
print('Recall: {0}'.format(recall_score(y_test_oot, clf_9.predict(x_test_oot[top_9]))))
print('F1_score: {0}'.format(f1_score(y_test_oot, clf_9.predict(x_test_oot[top_9]))))
print('Accuracy: {0}'.format(accuracy_score(y_test_oot, clf_9.predict(x_test_oot[top_9]))))


# In[58]:


# Save our trained model 
pkl_filename = "catboost_TOP_9_var.pkl"
with open (pkl_filename, 'wb') as file:
    pickle.dump(clf_9,file)


# In[ ]:





# ## PSI calculation in case when TARGET = 1

# In[59]:


data_1 = data[data['target_vsp']==1] # Train when target_vsp = 1


# In[60]:


oot_test_1 = OOT_test[OOT_test['target_vsp']==1] # OOT Test when target_vsp = 1


# In[61]:


x_test_oot_1 = oot_test_1.drop(['target_vsp', 'id_target'], axis=1)
y_test_oot_1 = oot_test_1.target_vsp


# In[62]:


x_1 = data_1.drop(['target_vsp', 'id_target'], axis=1)
y_1 = data_1.target_vsp


# In[65]:


x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_1[important_var], y_1, test_size = 0.3, random_state=0)


# In[66]:


for i in top_11:
    print(i)
    print(usr_al_psi(x_train_1[i], x_test_oot_1[i], 10))
    print()


# # График OOT

# In[22]:


# Upload model from pickle file
pkl_filename = "catboost_TOP_9_var.pkl"
with open(pkl_filename, 'rb') as file:
    clf_9 = pickle.load(file)


# In[1]:


N = 10
gini_score_cat = []
for i in range(0, N):
    top_10 = top_10[:i]
#    clf_10.fit(x_test_oot[top_10], y_test_oot, verbose=0)
    print(str(i+1)+' -> ROC AUC: {0} _//////_ GINI: {1}'.format(
        round(roc_auc_score(y_test_oot,
                            clf_9.predict_proba(x_test_oot[top_10])[:,1]),4),
        round(-1+2*roc_auc_score(y_test_oot,
                            clf_9.predict_proba(x_test_oot[top_10])[:,1]),4)))
    print(FI['Variable'].to_list()[i])
    print('')
    gini_score_cat.append(-1+2*roc_auc_score(y_test_oot,
                            clf_9.predict_proba(x_test_oot[top_10])[:,1]))


# In[76]:


plt.plot(list(range(len(gini_score_cat))), gini_score_cat)
plt.axis([0,53,0.1,.8])
plt.xlabel('N factors')
plt.ylabel('GINI score')
plt.show()


# # 6.Model Report <a name="p6"></a>

# In[17]:


import xlsxwriter
import datetime
import os

import sys


# In[7]:


import xlrd


# In[8]:


xlrd.__version__


# In[2]:


xlsxwriter.__version__


# In[18]:


df_discr = pd.read_excel('top_features.xlsx', header=0)


# In[19]:


df_discr


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x[df_discr['Variable'].to_list()], y, test_size = 0.3, random_state=0)


# In[ ]:


clf_


# In[54]:


path = 'Report.xlsx'
if os.path.exists(path):
    os.remove(path)
writer = pd.ExcelWriter(path, engine='xlsxwriter')


# In[47]:


# Рассчёт IV для одного фактора (array)

n_bins = 20
def iv_calc(x, y, bins=n_bins):
    if x.dtypes != 'object':
        df_for_iv = pd.DataFrame({'X': x, 'Y' : y})
        df_counts = pd.DataFrame({'bin': list(range(n_bins+1))})
        df_for_iv['bin'] = pd.qcut(df_for_iv.X.rank(method='first'), q=n_bins, labels=list(range(n_bins)))
        df_for_iv['bin']=df_for_iv['bin'].apply(lambda x: x+1)
        df_counts = pd.merge(df_counts, df_for_iv.groupby(by='bin', axis=0).count().Y, left_index=True, right_index=True, how='left')
        df_counts = pd.merge(df_counts, df_for_iv[df_for_iv['Y']==1].groupby(by='bin', axis=0).count().Y, left_index=True, right_index=True, how='left')
        df_counts = pd.merge(df_counts, df_for_iv[df_for_iv['Y']==0].groupby(by='bin', axis=0).count().Y, left_index=True, right_index=True, how='left')
        df_counts.columns = ['N bin', 'Count', '#Event', '#Non Event']
        df_counts[['Count', '#Event', '#Non Event']] = df_counts[['Count', '#Event', '#Non Event']].fillna(0)
        df_counts["%Event"] = df_counts['#Event']/df_for_iv[df_for_iv.Y==1].shape[0]
        df_counts["%Non Event"] = df_counts['#Non Event']/df_for_iv[df_for_iv.Y==0].shape[0]
        if (df_counts[df_counts["%Event"]==0].shape[0] != 0):
            df_counts[df_counts["%Event"]==0] = df_counts[df_counts["%Event"]==0].apply(lambda x: x + 0.001)
        if (df_counts[df_counts["%Non Event"]==0].shape[0] != 0):
            df_counts[df_counts["%Non Event"]==0] = df_counts[df_counts["%Non Event"]==0].apply(lambda x: x + 0.001)
        df_counts["WOE"] = np.log(df_counts["%Event"]/df_counts["%Non Event"])
        df_counts["%Event - %Non Event"] = df_counts["%Event"] - df_counts["%Non Event"]
        df_counts["IV"] = df_counts["%Event - %Non Event"]*df_counts["WOE"]
        IV_sum = round(df_counts.IV.sum(), 4)
    else:
        print('OBJECT')
    return IV_sum


# In[59]:


# Запись в отчёт IV по всем нужным факторам

def iv_sheet(x, y):

    df_iv_all = pd.DataFrame({'Variable': [], 'IV': [], 'Selected': [], 'Description': []})
    for i in list(x.columns):
        print(i)
        df_iv_var = pd.DataFrame({'Variable': [i]})
        df_iv_var['IV'] = iv_calc(x[i], y)
        if i in df_discr['Variable'].to_list():
            df_iv_var['Selected'] = 'Y'
            df_iv_var['Description'] = df_discr[df_discr['Variable']==i]['Description'].values
        else:
            df_iv_var['Selected'] = 'N'
            df_iv_var['Description'] = '-'
        df_iv_all = pd.concat([df_iv_all, df_iv_var])
    df_iv_all = df_iv_all.sort_values(by='Selected', ascending=False)
    
    # сохранение в Отчёт!
    df_iv_all.to_excel(writer, sheet_name='Variables_IV', index=False)
    df_iv_all.to_csv('Variables_IV.csv',index=False)
#    show_thms = pd.read_excel('Report.xlsx', header=0, sheet_name='Variables_IV')
#    return show_thms 


# In[60]:


# Вызов функций для записи!

iv_sheet(x[obj_var+num_var], y)


# In[41]:


# Создание листа с аналитикой для Train/Validation/OOT и запись в отчёт!

def stats_for_predict(y_f, y_2f, x_f, x_predict_f, sheet_name):
    
    clf_best = CatBoostClassifier(iterations=100, max_depth=9, scale_pos_weight=35, random_seed=0) # <-- Put in your hyper params here!!!
    clf_best.fit(x_f, y_f)
    df_helper = pd.DataFrame({'y_score': clf_best.predict_proba(x_predict_f)[:,1],
                      'y_true': y_2f})
    df_helper['Bin'] = pd.qcut(df_helper.y_score.rank(method='first'), q=20, labels=list(range(20)))
    df_full_st = df_helper.groupby('Bin').apply(np.max)['y_score'].rename('Prob_max')
    df_full_st = pd.merge(df_full_st, df_helper.groupby('Bin').apply(np.min)['y_score'].rename('Prob_min'),
                      how='left', left_index = True, right_index = True)
    df_full_st = pd.merge(df_full_st, df_helper.groupby('Bin').apply(np.mean)['y_score'].rename('Prob_avg'),
                      how='left', left_index = True, right_index = True)
    df_full_st = pd.merge(df_full_st, df_helper.groupby('Bin').size().rename('# obs'),
                      how='left', left_index = True, right_index = True)
    df_full_st = pd.merge(df_full_st, df_helper[df_helper['y_true']==1].groupby('Bin').size().rename('# event'),
                      how='left', left_index = True, right_index = True)
    df_full_st = pd.merge(df_full_st, df_helper[df_helper['y_true']==0].groupby('Bin').size().rename('# nonevent'),
                      how='left', left_index = True, right_index = True)
    df_full_st = pd.merge(df_full_st, (df_full_st['# event']/df_full_st['# obs']).rename('event rate'),
                      how='left', left_index = True, right_index = True)
    df_full_st.reset_index(inplace=True)
    df_full_st.sort_values(by='Bin', axis=0, ascending=False, inplace=True)
    df_full_st.set_index('Bin', inplace=True)
    df_full_st['cum_#_ev'] = df_full_st['# event'].cumsum()
    df_full_st['cum_#_nonev'] = df_full_st['# nonevent'].cumsum()
    df_full_st['1-Specificity'] = df_full_st['# nonevent'].cumsum() / df_full_st['# nonevent'].sum()
    df_full_st['Sensetivity'] = df_full_st['# event'].cumsum() / df_full_st['# event'].sum()
    temp_arr = ['-']*19
    temp_arr.append(
        str(round(-1+2*roc_auc_score(y_2f, clf_best.predict_proba(x_predict_f)[:,1]),4)))
    df_full_st['Gini'] = temp_arr
    temp_arr = ['-']*19
    temp_arr.append(
        str(round(roc_auc_score(y_2f, clf_best.predict_proba(x_predict_f)[:,1]),4)))
    df_full_st['AUC'] = temp_arr
    df_full_st['BaseLine'] = np.linspace(0,1,21)[1:]
    
    df_full_st['Prob_max'] = df_full_st['Prob_max'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st['Prob_min'] = df_full_st['Prob_min'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st['Prob_avg'] = df_full_st['Prob_avg'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st['event rate'] = df_full_st['event rate'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st['1-Specificity'] = df_full_st['1-Specificity'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st['Sensetivity'] = df_full_st['Sensetivity'].apply(lambda x: '{:.2%}'.format(x))
    df_full_st.reset_index(inplace=True)

    df_full_st.to_excel(writer, sheet_name=sheet_name, index=False)


# In[42]:


# Вызов функций для записи!
start = datetime.now()
print(start)

stats_for_predict(y_train, y_train, x_train, x_train, 'KS Gini Train')
stats_for_predict(y_train, y_test, x_train, x_test, 'KS Gini Val')
stats_for_predict(y_train, y_test_oot, x_train, x_test_oot, 'KS Gini OOT')

stop = datetime.now()
print(stop)
print('Upload Time: ',stop-start)


# In[ ]:


writer.save()


# In[ ]:





# In[ ]:




