# coding:utf-8
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from MyFuck import readFromDBSql
from MyFuck import exeSql
from MyFuck import writeMany2DB
import pandas as pd
from pandas import DataFrame
import numpy as np

table_name = 'cut_lv_article_search'

db_info=('123.59.50.66','root','T1a2b3l0e1a*u!',3306,'test_hwli',"utf8")
sql_delete = 'DROP TABLE IF EXISTS %s ;' % table_name
exeSql(db_info,sql_delete)

sql_create ='''
CREATE TABLE `%s` (
  `id` int(20) NOT NULL AUTO_INCREMENT COMMENT '序号',
  `word` varchar(100) DEFAULT NULL COMMENT '分词单词',
  `tfidf` double DEFAULT NULL COMMENT 'tf-idf词频列求和',
  `tf` double DEFAULT NULL COMMENT 'tf词频列求和',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=452723 DEFAULT CHARSET=utf8 ROW_FORMAT=COMPACT COMMENT='分词结果';
''' % table_name

exeSql(db_info,sql_create)
fields =('word','tfidf','tf')
sql_shop = 'select content from tmp_rdg_lv_article ;'

def work():
    shoppings = readFromDBSql(db_info,sql_shop)
    print len(shoppings)
    n = 0
    corpus =[]
    for shop in shoppings:
        print 'n=========',n
        n +=1
        if shop[0]:
            seg_list = jieba.cut_for_search(shop[0]) 
            cut_str = " ".join(seg_list)
            corpus.append(cut_str)
    
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第
    tfidf_sum0 = np.asarray(tfidf.sum(axis =0))[0,:]
    tf=vectorizer.fit_transform(corpus)
    tf_sum0 = np.asarray(tf.sum(axis =0))[0,:]
    print 'tf~~~~~~~~~~~',tf 
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    
    result = np.c_[np.array(word),tfidf_sum0,tf_sum0].tolist()
    writeMany2DB(db_info,fields,table_name,result)

if __name__ == "__main__":
    work()
