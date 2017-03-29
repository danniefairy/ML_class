import numpy as np
import sys
import csv
import pickle
#necessary for jupyter
%matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt

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
    predict="./result/res.csv"
    
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
    W=[]
    b=[]
    for i in range(length+1):
        W.append(np.random.random(length)/1000000)
        b.append(0.000001)
    return np.array(W),np.array(b)


def sigmoid(X):
    return 1/(1+np.exp(np.dot(-1,X)))


def training(X,y,W,b):
    for rounds in range(100):
        #accuracy
        zzz=[]
        for s in range(len(W)-1):
            zzz.append(np.dot(X,W[s])+b[s])
        aaa=sigmoid(zzz)
        out=np.dot(np.transpose(aaa),W[len(W)-1])+b[len(b)-1]
        out=sigmoid(out)
        #copy for acc and loss
        a_out=np.copy(out)
        a_out[a_out<0.5]=0
        a_out[a_out>=0.5]=1
        acc=(1-np.logical_xor(a_out,np.transpose(y)[0])).sum()/len(y)
        Loss=((out-np.transpose(y)[0])**2).sum()
        print(rounds,":","accuracy:",acc,"Loss:",Loss)

        
        
        #first layer
        a=[]
        for i in range(len(X)):
            z=[]
            for j in range(len(W)-1):
                z.append(np.dot(X[i],W[j])+b[j])
            a.append(sigmoid(z))

        
        #second layer
        #a=32561*106 || 32561times update 106 features
        a=np.array(a)
        dldz_second=[]
        zz=[]
        g_second_pre=0.001
        b_second_pre=0.001
        eta=0.01
        for k in range(len(a)):
            #np isn't iterable
            zz.append((np.dot(a[k],W[len(W)-1])+b[len(b)-1]).tolist())
            aa=(sigmoid(zz[k]))
            #aa=1*1

            #second layer update
            dldz_second.extend(y[k]-aa)
            dz_seconddw_second=np.reshape(np.transpose(a[k]),(len(a[k]),-1))

            g_second=-np.dot(dz_seconddw_second,dldz_second[k])
            g_second_pre+=g_second**2
            
            b_second=-dldz_second[k]
            b_second_pre+=b_second**2

            W[len(W)-1]-=np.transpose(eta*g_second/np.sqrt(g_second_pre))[0]
            b[len(b)-1]-=np.transpose(eta*b_second/np.sqrt(b_second_pre))

        #update first layer weight   
        #W[len(W)-1] 106,dldz_second 32561
        pre=0.0001
        pre_b=0.0001
        for r in range(len(zz)):
            for w in range(len(W)-1):
                diff_sig_dldz_w_x=-sigmoid(zz[r])*(1-sigmoid(zz[r]))*dldz_second[r]*W[len(W)-1][w]*X[r]
                diff_sig_dldz_b_x=-sigmoid(zz[r])*(1-sigmoid(zz[r]))*dldz_second[r]*W[len(W)-1][w]
                
                pre+=diff_sig_dldz_w_x**2
                pre_b+=diff_sig_dldz_b_x**2
                
                W[w]-=eta*diff_sig_dldz_w_x/np.sqrt(pre)
                b[w]-=eta*diff_sig_dldz_b_x/np.sqrt(pre_b)
                
            if(r%10000==0):
                print(rounds,":",r)
        #save W
        with open("deep.pickle",'wb') as file:
                model=[W]
                pickle.dump(model,file)
        
        #read W
        #with open('deep.pickle','rb') as file:
                #model=pickle.load(file)
                #W=(model[0])
        

        
    
def main():
    #data extraction
    train_data,target_data,test_data,predict=data_extraction()
    
    #data normalization
    train_data,test_data=normalization(train_data,test_data)
    
    #weight generation
    W,b=weight(len(train_data[0]))

    #training
    W=training(X=train_data,y=target_data,W=W,b=b)
    
    
    '''
    #accuracy
    zzz=[]
    for s in range(len(W)-1):
        zzz.append(np.dot(test_data,W[s]))
    aaa=sigmoid(zzz)
    a_out=np.dot(np.transpose(aaa),W[len(W)-1])
    a_out[a_out<0.5]=0
    a_out[a_out>=0.5]=1
    #output csv
    data=[]
    data.append(['id','label'])
    for k in range(len(a_out)):
        data.append([k+1,int(a_out[k])])
    f = open("test_deep","w")
    w = csv.writer(f)
    w.writerows(data)
    f.close()
    print("finish!")
    '''
    
if __name__=="__main__":   
    main()