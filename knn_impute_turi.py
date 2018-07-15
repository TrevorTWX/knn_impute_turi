import turicreate as tc
import numpy as np
import pandas as pd
from scipy import stats
import tqdm


# In[36]:


def get_nan_col(table):
    
    '''
    This method will get all the columns with None values
    in the given table.
    
    table:    should be turicreate.SFrame object
    
    return:   array of column names with missing values
    '''
    
    #convert to pandas.dataframe for checking None's
    #Don't know how to do it with turicreate.SFrame
    #ANY Suggestion is WELCOME!
    nan_col = []
    table_pd = table.to_dataframe()
    table_pd = table_pd.drop(['X1'],axis=1)
    for i in range(1,table_pd.shape[1]):
        if table_pd.iloc[:,i].isnull().values.any():
            nan_col.append(table_pd.iloc[:,i].name)
    return nan_col


# In[37]:


def knn_impute_turi_col(target, attributes, knn):
    
    '''
    This method will replace the None values in the column 'target'
    using KNN on the attributes set with K=k.
    
    For numerical values, the missing value is filled with the mean of the neighbors'.
    For other kinds of values, the missing value is filled with the mode of the neighbors'.
    
    target:  Target column, can be in any 2d-array form, ie. SArray, np array, list...
    attributes:   a table with NO MISSING VALUES, should be the same number of rows as target
    k:  K nearest neighbors
    
    NOTICE: the k, K nearest neighbors, is not precisely followed. 
            Some columns may have lots of missing values, finding exactly N nearest neighbors can be horrible.
            If 0 < neighbors_with_no_missing_values_in_target_column < k for K neighbors, the found numbers of non-missing values will be used
            If all k neighbors having Nane for target column, will expand N, till ONE neighbor with non-missing value is found
    
    return:  a new target list with no missing values.
    '''
    
    target = list(target)
    knn_model = tc.nearest_neighbors.create(attributes, verbose = False)
    for i in tqdm.tqdm_notebook(range(0,len(target))):
        if target[i] == None:
            query_result = knn_model.query(attributes[i:i+1], k=knn, verbose = False)
            ref_label = query_result['reference_label']
            to_cal = []
            for index in ref_label:
                if target[index] != None:
                    to_cal.append(target[index])
            additional = 0
            while len(to_cal)<1:
                additional = additional + 1
                query_result = knn_model.query(attributes[i:i+1], k = knn+additional, verbose = False)
                ref_label = query_result['reference_label']
                to_cal = []
                for index in ref_label:
                    if target[index] != None:
                        to_cal.append(target[index])
            if isinstance(to_cal[0], float):
                new_value = np.mean(to_cal)
            else:
                new_value = stats.mode(to_cal)[0][0]
            target[i] = new_value
    return target


# In[38]:


def knn_impute_turi(sframe,k):
    
    '''
    This method will fill all the missing values in the SFrame using 
    KNN. The whole process is designed using Turi Create to handel large
    dataset that cannot be fit into RAM.
    
    sframe:  a turicreate.SFrame object
    
    k:  K nearest neighbors for KNN
    
    return:  sframe without any missing value
    
    NOTICE: The whole set is required to have at least one column without
            any missing value. However, it is recommanded that more than 
            half of the columns should be without any missing value.
            
    NOTICE: The order of the rows is preserved but the order of columns is not.
    NOTICE: Please do not include any feature that is identical for each entry,
            ie. ID_NUMBERS...
    '''
    
    knn_cleaned_cols = tc.SFrame()
    nan_col = get_nan_col(sframe)
    attributes = sframe.remove_columns(nan_col)
    for i in tqdm.tqdm_notebook(range(0,len(nan_col))):
        target = list(sframe[nan_col[i]])
        knn_cleaned_cols = knn_cleaned_cols.add_column(tc.SArray(knn_impute_turi_col(target,attributes,k)),column_name = nan_col[i])
    return attributes.add_columns(knn_cleaned_cols)
