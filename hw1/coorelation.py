import sys
import csv
import numpy as np
import pickle
%matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing


def main():
    #train=sys.argv[1]
    #test=sys.argv[2]
    #result= sys.argv[3]
    train="./data/train.csv"
    test="./data/test_X.csv"

    
    f = open(train, 'r', encoding='ISO-8859-1')
    excel_raw=[]
    for row in csv.reader(f):
        excel_raw.append(row)
    #extract raw data from excel
    excel_raw=np.array(excel_raw)
    train_raw=np.array(excel_raw[1:,3:])
    train_raw=np.where(train_raw =='NR', 0, train_raw)
    train_raw=train_raw.astype(np.float)
    day=np.split(train_raw,240)

    #coorelation
    a=[]
    for d in range(240):
        for i in range(18):
            if(d==0):
                a.append(abs(np.corrcoef(day[d][i],day[d][9])[1,0]))
            else:
                if(np.corrcoef(day[d][i],day[d][9])[1,0]>=(-1) and np.corrcoef(day[d][i],day[d][9])[1,0]<=1):
                    a[i]+=((np.corrcoef(day[d][i],day[d][9])[1,0]))#"abs" or not
                else:
                    a[i]=0
    
    t=np.arange(0,18,1)
    plt.plot(t,a)


if __name__=="__main__":
    main()
    
