import numpy as np
import sys
import csv
import pickle


def data_extraction():
    #raw_train=sys.argv[1]
    #raw_test=sys.argv[2]
    #train= sys.argv[3]
    #test=sys.argv[4]
    #test=sys.argv[5]
    #predict=sys.argv[6]
    train="./data/X_train.csv"
    target="./data/Y_train.csv"
    test="./data/X_test.csv"
    predict="./result/log_res.csv"
    
    #extract training data
    f = open(train, 'r')
    line_count=0
    train_data=[]
    for row in csv.reader(f):
        if(line_count==0):
            pass
        else:
            train_data.append(row)
        line_count+=1
    f.close()
    
    #extract label data
    f = open(target, 'r')
    target_data=[]
    for row in csv.reader(f):
        target_data.append(row)
    f.close()
    
    #extract testing data
    f = open(test, 'r')
    line_count=0
    test_data=[]
    for row in csv.reader(f):
        if(line_count==0):
            pass
        else:
            test_data.append(row)
        line_count+=1
    f.close()
    
    train_data=np.array(train_data).astype(np.float)
    target_data=np.array(target_data).astype(np.float)
    test_data=np.array(test_data).astype(np.float)
    return train_data,target_data,test_data,predict


def normalization(train,test):
    total=train.tolist()
    total.extend(test.tolist())
    total=np.array(total)
    
    t=np.transpose(total)
    for i in range(len(t)):
        t[i]=(t[i]-t[i].mean())/t[i].std()
    total=np.transpose(t)
    #train_data,test_data
    return total[:len(train)],total[len(train):]
    
    
def weight(length):
    return np.random.random(length)/1000000,0.000001


def sigmoid(X):
    return 1/(1+np.exp(np.dot(-1,X)))


def training(X,y,W,b):
    W=np.reshape(W,(len(W),-1))

    g_pre=0.000005
    g_b_pre=0.000005
    eta=1
    for i in range(5000):
        for j in range(len(y)):
            z=np.dot(X[j],W)+b
            #reshape a into column
            a=np.reshape(sigmoid(z),(len(z),-1))
            #update
            g=-np.dot(np.reshape(np.transpose(X[j]),(len(X[j]),-1)),y[j]-a) 
            g_b=-(y[j]-a)
            g_pre+=g**2
            g_b_pre+=g_b**2

            W=W-eta*g/np.sqrt(g_pre)
            b=b-eta*g_b/np.sqrt(g_b_pre)
            
        if(i%10==0):
            zz=np.dot(X,W)+b
            #reshape a into column
            aa=np.reshape(sigmoid(zz),(len(zz),-1))
            #accuracy
            c = np.copy(aa)
            c[c>=0.5]=1
            c[c<0.5]=0
            #!xor
            acc=(1-np.logical_xor(c,y)).sum()/len(c)
            print("acc:",acc,"RMSE:",np.sqrt(((aa-y)**2).sum()/len(aa)))
            
            #save W
            with open("sgd.pickle",'wb') as file:
                model=[W,b]
                pickle.dump(model,file)
            
            if(acc>0.8533):
                break

    
    
    return W,b
        

def test(W,b,test_data,predict):
    z=np.dot(test_data,W)+b
    a=sigmoid(z)
    a[a<0.5]=0
    a[a>=0.5]=1
    #output csv
    data=[]
    data.append(['id','label'])
    for k in range(len(a)):
        data.append([k+1,int(a[k][0])])
    f = open(predict,"w")
    w = csv.writer(f)
    w.writerows(data)
    f.close()
    print("finish!")
    
    
def main():
    #data extraction
    train_data,target_data,test_data,predict=data_extraction()
    
    #data normalization
    train_data,test_data=normalization(train_data,test_data)
    
    #weight generation
    W,b=weight(len(train_data[0]))

    #training
    #W,b=training(X=train_data,y=target_data,W=W,b=b)
    
    #read W
    with open('sgd.pickle','rb') as file:
            model=pickle.load(file)
            W=(model[0])
            b=(model[1])
    
    #output csv
    test(W,b,test_data,predict)

if __name__=="__main__":   
    main()