# knn_impute_turi
Filling missing values in big dataset using KNN with Turi Create. (For dataset/operation size cannot fit into RAM)

This program is inspired by Yohan Obadia, https://towardsdatascience.com/the-use-of-knn-for-missing-values-cf33d935c637
The purpose of building this program is that the original codes by Yohan Obadia is based on Pandas and will cause RAM full for big dataset. Yohan Obadia's code can be found here: https://gist.github.com/YohanObadia/b310793cd22a4427faaadd9c381a5850

Details for the main method:
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
            
For information about Turi Create, visit: https://github.com/apple/turicreate
