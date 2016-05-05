#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
Created on Dec 14, 2015
PTEの学習結果をプロットする

@author: ochi
'''

import sys
import csv
import codecs
#import math
import codecs
#import pickle
import numpy as np
import scipy.spatial.distance as spd

import sklearn.linear_model as sl
import sklearn.manifold as sm

import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

corpidfile = "../data20160226/corpid.csv"
impidfile  = "../data20160226/impid.csv"
stationidfile  = "../data20160226/secret/station2.1u.tsv"

#pte_resultfile = "vec_2nd_wo_norm.txt" # PTEの学習結果ファイル
#pte_resultfile = "vec_2nd_wo_norm_div10k.txt" # PTEの学習結果ファイル
#pte_resultfile = "vec_2nd_wo_norm.txt0" # PTEの学習結果ファイル
tSNEFile = "tsneplot.csv"

bestN = 0

#dw_resultfile = "deepwalk.embeddings" # DeepWalkの学習結果ファイル
#label_csvfile = "impressionlabeling_binary.csv" # Excel上でimpressionlabeling_select.csvから作成

#newStationIdNameDicFile = "newStationIDNameDic.pickle"
#newStationNameIdDicFile = "newStationNameIDDic.pickle"

def makeIdNameDicFromCSV(filename,delimiter=',',header=True):
    a_dic = {}
    f = codecs.open(filename, 'r', "utf_8")
    reader = csv.reader(f, delimiter=delimiter)
    for i, row in enumerate(reader):
        if i == 0 and header is True:
            # 最初の行はヘッダ
            continue
        else:
            new_id  = row[0]
#            print new_id
            a_val = str(row[1])
            a_dic[new_id] = a_val
    f.close()

    return a_dic

def makeIdNameDicFromCSVstation(filename,delimiter=',',header=True):
    a_dic = {}
    b_dic = {}
    f = codecs.open(filename, 'r', "utf_8")
    reader = csv.reader(f, delimiter=delimiter)
    for i, row in enumerate(reader):
        if i == 0 and header is True:
            # 最初の行はヘッダ
            continue
        else:
            new_id  = str(row[0])
#            print new_id
            a_val = str(row[2])+"・"+str(row[1])
            a_dic[new_id] = a_val
            b_dic[new_id] = [str(row[1]),str(row[2])]#stName, corpN
    f.close()

    return a_dic,b_dic

def getIdVec(pte_resultfile):
    idVecDic = {}
    f = codecs.open(pte_resultfile, 'r', "utf_8")
    reader = csv.reader(f, delimiter=' ')
    for i, row in enumerate(reader):
        if i == 0:
            # 最初の行はヘッダ
            continue
        else:
#            print i
#            print row

            new_id  = str(row[0])
#            print new_id
#            print row[1:]
            #なぜか一番後ろに空白があるんで
            a_vec = np.array(row[1:-1], dtype=np.float)
            idVecDic[new_id] = a_vec
    f.close()

    return idVecDic

def getNearest(a_id, idVecDic, idcorpdic, idimpdic, bestN=10):
    a_vec = idVecDic[a_id]
    idDistanceDic = {}
    ids = idVecDic.keys()
    for b_id in ids:
        if b_id == a_id:
            continue
        else:
            b_vec = idVecDic[b_id]
            dist = spd.euclidean(b_vec,a_vec)
            idDistanceDic[b_id] = dist

    nears = []
    for k,v in sorted(idDistanceDic.items(), key=lambda x: x[1]):
#        if k not in idcorpdic.keys() and k not in idimpdic.keys():
#            nears.append(k)
        nears.append(k)
    return nears[:bestN]

def run(pte_resultfile, an_id):
    print "pte_resultfile:%s"%(pte_resultfile)
    print "an_id:%s"%(an_id)
    # csv読み込み
    idcorpdic     = makeIdNameDicFromCSV(corpidfile)
#    print "idcorpdic"
#    print idcorpdic
#    print "a"
    idimpdic      = makeIdNameDicFromCSV(impidfile, delimiter='\t')
#    print "idimpdic"
#    print idimpdic
#    print "b"
    idstationdic, idstationdic1  = makeIdNameDicFromCSVstation(stationidfile, delimiter='\t')
#    print "idstationdic"
#    print idstationdic
#    print "c"

    idNameDic = {}
    idNameDic.update(idcorpdic)
    idNameDic.update(idimpdic)
    idNameDic.update(idstationdic)
#    print idNameDic

    # PTEの学習結果のベクトル取得
    idVecDic = getIdVec(pte_resultfile)

    # impressionの近傍のIDを取得
    new_ids = idVecDic.keys()
    corps = []
    for a_id in new_ids:
        if a_id in idstationdic1.keys():
            corp = idstationdic1[a_id][1]
        else:
            corp = None

        corps.append(corp)

#    print new_ids
    plotOrNot = [False]*len(new_ids)
#    print idcorpdic
#    print new_ids
    for a_id in idcorpdic.keys():
        if a_id in new_ids:
            idx = new_ids.index(a_id)
            plotOrNot[idx] = True

    for a_id in idimpdic.keys():
#        print a_id
#        print type(a_id)
#        print type(new_ids[0])
#        for i, new_id in enumerate(new_ids):
#            print "%d:%s"%(i,new_id)
#            print type(new_id)
        if a_id in new_ids:
            idx = new_ids.index(a_id)
        else:
            continue
    
        plotOrNot[idx] = True
        nears = getNearest(a_id, idVecDic, idcorpdic, idimpdic, bestN=bestN)
        print "%sに近いベクトル"%(idimpdic[a_id])
        for i,near in enumerate(nears):
            idx = new_ids.index(near)
            print "%d, %s"%(i+1, idNameDic[near])
            plotOrNot[idx] = True
        print ""


    # 獲得した特徴量の可視化(TSNE method)
#    for new_id in new_ids:
#        print new_id

    names = [ idNameDic[new_id] for new_id in new_ids ]
    X = np.array(idVecDic.values())

    if X.shape[1] == 2:
        print "vector dimension is 2, so dont use TSNE"
        resX = X
    else:
        model = sm.TSNE(n_components=2, random_state=0)
        resX = model.fit_transform(X)

    draw2DScatterPlot(new_ids, corps, names,resX,plotOrNot,"PTE_Result", out_header="out", an_id=str(an_id))


def draw2DScatterPlot(new_ids, corps, vocab, reduce_vecs, plotOrNot, parameterStr, out_header="out", an_id=""):
    colors = ["b","g","c","m","y","w"]
    set_corps = list(set(corps))
    print set_corps

    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    # 軸ラベルの設定
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")

    # 表示範囲の設定
    min_val_x = np.min(reduce_vecs[:,0]) -0.1
    min_val_y = np.min(reduce_vecs[:,1]) -0.1
    max_val_x = np.max(reduce_vecs[:,0]) +0.1
    max_val_y = np.max(reduce_vecs[:,1]) +0.1
    print min_val_x
    print max_val_x
    print min_val_y
    print max_val_y

#    ax.set_xlim(min_val_x,max_val_x)
#    ax.set_ylim(min_val_y,max_val_y)
    ax.set_xlim(-40,40)
    ax.set_ylim(-40,70)

    #plot
#    ax.plot(reduce_vecs[:,0], reduce_vecs[:,1], "o", color="#cccccc", ms=4, mew=0.5)

    f = open(tSNEFile, "w")
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
#    plotindices = []
    for i, w in enumerate(vocab):
        xy = reduce_vecs[i,:]
        writer.writerow([w,reduce_vecs[i,0],reduce_vecs[i,1]])
#        ax.plot(reduce_vecs[i,0], reduce_vecs[i,1], "o", color="#cccccc", ms=4, mew=0.5)
#        ax.annotate(w,xy=xy, xytext=(1,1), textcoords="offset points")
        if plotOrNot[i] is True:
#            plotindices.append(i)
#            continue 
            ax.plot(reduce_vecs[i,0], reduce_vecs[i,1], "o", color="#ff0000", ms=4, mew=0.5)
            ax.annotate(vocab[i],xy=xy, xytext=(1,1), textcoords="offset points")
#        if new_ids[i] in  set(["0860","0834","0513","0962"]) or plotOrNot[i] is True:
        else:
            corpName = corps[i]
#            print corpName
            corp_idx = set_corps.index(corpName)
            color = colors[corp_idx%len(colors)]
#            print color
#            ax.plot(reduce_vecs[i,0], reduce_vecs[i,1], "o", color="#cccccc", ms=4, mew=0.5)
            ax.plot(reduce_vecs[i,0], reduce_vecs[i,1], "o", color=color, ms=8, mew=0.2)
#            if i % 15 == 0: 
#                ax.annotate(w,xy=xy, xytext=(1,1), textcoords="offset points")

#        if i % 15 == 0: 
#            ax.annotate(w,xy=xy, xytext=(1,1), textcoords="offset points")
#        else:
#            ax.annotate('',xy=xy, xytext=(1,1), textcoords="offset points")

#    for i in plotindices:
#        xy = reduce_vecs[i,:]
#        ax.plot(reduce_vecs[i,0], reduce_vecs[i,1], "o", color="#ff0000", ms=4, mew=0.5)
#        ax.annotate(vocab[i],xy=xy, xytext=(1,1), textcoords="offset points")

    pyplot.axis("off")
#    pyplot.show()
    f.close()
    pyplot.savefig("pic_"+out_header+ an_id + ".png")
#    pyplot.savefig("pic_"+out_header+ an_id + ".pdf")
#    pyplot.savefig("pic"+parameterStr+".png")
    pyplot.clf()
    pyplot.close()

    return

if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    run(arg1,arg2)
