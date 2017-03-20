import sys
import csv
import numpy as np
import pickle
#necessary for jupyter
#%matplotlib notebook
#import matplotlib
#import matplotlib.pyplot as plt
from sklearn import preprocessing


def main(eta,output_pickle,train_condition):
    #=================File loacation=================
    train=sys.argv[1]
    test=sys.argv[2]
    result_csv= sys.argv[3]
    #train="./data/train.csv"
    #test="./data/test_X.csv"
    #=================End of File loacation=================
        
    
    #=================Data extraction=================
    Data = []
    for i in range(18):
        Data.append([])

    n_row = 0
    text = open(train, 'r', encoding='ISO-8859-1') 
    row = csv.reader(text , delimiter=",")
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != "NR":
                    Data[(n_row-1)%18].append( float( r[i] ) )
                else:
                    Data[(n_row-1)%18].append( float( 0 ) )	
        n_row =n_row+1
    text.close()

    train_x = []
    train_y = []

    for i in range(12):
        for j in range(471):
            train_x.append(  [1] )
            for t in range(18):
                for s in range(9):
                    train_x[471*i+j].append( Data[t][480*i+j+s] )
            train_y.append( Data[9][480*i+j+9] )
            train_x[471*i+j]=train_x[471*i+j][1:]


    #=================End of Data extraction=================

    #=================Elimitate low coorelative feature from training set=================
    for i in range(5652):
        train_x[i]=np.delete(train_x[i], [9,10,11,12,13,14,15,16,17])#1*
        train_x[i]=np.delete(train_x[i], [27,28,29,30,31,32,33,34,35])#4
        train_x[i]=np.delete(train_x[i],[72,73,74,75,76,77,78,79,80])#10
        #train_x[i]=np.delete(train_x[i],[81,82,83,84,85,86,87,88,89])#10*
        #pass

    #eliminate element contain negative value of pm2.5
    train_x=np.array(train_x)
    temp=[]
    targ=[]
    for i in range(5652):
        if(len(np.where(train_x[i][63:72]<0)[0])==0):#72:81  63:72
            temp.append(train_x[i])
            targ.append(train_y[i])
    train_x=temp#5504
    train_y=targ
    #train target size: 5504
    #=================End of Elimitate low coorelative feature=================

    #=================Testing data extraction=================
    f = open(test, 'r', encoding='ISO-8859-1')
    excel_raw=[]
    for row in csv.reader(f):
        excel_raw.append(row)

    excel_raw=np.array(excel_raw)
    test_raw=np.array(excel_raw[:,2:])
    test_raw=np.where(test_raw =='NR', 0, test_raw)
    test_raw=test_raw.astype(np.float)
    day_test=np.split(test_raw,240)

    #eliminate negative pm2.5
    for d in range(240):
        temp=0
        for i in range(9):
            if(day_test[d][9][i]>0):
                temp=temp+day_test[d][9][i]
        for i in range(9):
            if(day_test[d][9][i]<0):
                day_test[d][9][i]=temp/9

    #make output value positive
    pm_25=[]
    for i in range(240):
        pm_25.append(day_test[i][9][7:].mean())

    #delete low coorelation features
    for d in range(240):
            day_test[d]=np.delete(day_test[d], (1,4,10), axis=0)#4

    #make 9 hours data to 1d
    test=[]
    one_day=[]
    for days in range(240):
        test.append(np.reshape(day_test[days],(1,-1))[0])

    #=================End of Testing data extraction=================

    #=================Data normalization=================
    x=np.concatenate((train_x, test), axis=0)
    Len=int(len(train_x))
    #feature wise normalization
    t=np.transpose(x)
    #transpose train normalize
    for days in range(135):#144 135
        #t[days]=(t[days]-t[days].min())/(t[days].max()-t[days].min()) 
        t[days]=(t[days]-t[days].mean())/t[days].std()
    p=np.transpose(t)
    train_x,test=p[:5504],p[5504:]#split training data and testing data
    #=================End of Data normalization=================

    #=================Shuffle data=================
    #feature wise shuffle
    #np.random.seed(0)
    arr = np.arange(0,Len,1)
    np.random.shuffle(arr)

    #feature wise append shuffle
    train_shuffle=[]
    target_shuffle=[]
    for i in range(Len):
        train_shuffle.append(train_x[arr[i]])
        target_shuffle.append(train_y[arr[i]])
    #=================End of Shuffle data=================

    if(train_condition):
        #=================Weight initialization=================
        w_len=len(train_shuffle[0])
        W1=np.random.random(w_len)/10000000000
        W2=np.random.random(w_len)/10000000000
        W3=np.zeros(w_len)+0.00001
        W4=np.zeros(w_len)+0.00001
        W5=np.zeros(w_len)+0.00001
        bias=0.000000001
        #=================End of Weight initialization=================

        #=================Splite training set and validation set=================      
        #train
        input_data=train_shuffle[:5200]
        input_target=target_shuffle[:5200]

        #validation
        validation=np.array(train_shuffle[5200:])
        validation_target=np.array(target_shuffle[5200:])
        #=================End of Splite training set and validation set================= 


        #=================Parameter initialization================= 
        x=np.array(input_data)
        y=np.array(input_target)

        #x axis
        #t = np.arange(0.0, int(len(input_data)), 1)
        #plt.ion()
        N=int(len(x))
        batch=int(N/1)
        L_post=0
        L=100000000000
        preb=0
        pre1=0
        pre2=0
        m1=0
        m2=0
        G1=0
        G2=0
        change_th=91000
        #=================End of Parameter initialization=================

        #=================Training iteration================= 
        for i in range(500):
            for num in range(batch):
                #=================Batch size================= 
                start=int((N/batch)*num)
                end=int((N/batch)*(num+1))
                x_b=x[start:end]
                y_b=y[start:end]
                #=================End of Batch size================= 

                #=================Calculate gradient descent================= 
                z_b=np.dot(x_b**2,W2)+np.dot(x_b,W1)+bias
                g1=np.dot(((-1)*(np.transpose(x_b))),(y_b-z_b))
                g2=np.dot(((-1)*(np.transpose(x_b**2))),(y_b-z_b))
                b=sum(-1*(y_b-z_b))

                #adagrad
                pre1=pre1+g1**2
                pre2=pre2+g2**2
                preb=preb+b**2
                W1=W1-eta*(g1)/np.sqrt(pre1+0.0001)
                W2=W2-eta*(g2)/np.sqrt(pre2+0.0001)
                bias=bias-eta*b/np.sqrt(preb+0.0001)

                '''
                #RMSProp
                B=0.9
                pre1=pre1*B+(1-B)*g1**2
                pre2=pre2*B+(1-B)*g2**2
                preb=preb*B+(1-B)*b**2
                W1=W1-eta*(g1)/np.sqrt(pre1+0.0001)
                W2=W2-eta*(g2)/np.sqrt(pre2+0.0001)
                bias=bias-eta*b/np.sqrt(preb+0.0001)
                '''
                '''
                #adam
                m1=0.9*m1+0.1*g1
                m2=0.9*m2+0.1*g2
                G1=0.99*G1+0.01*(g1**2)
                G2=0.99*G2+0.01*(g2**2)
                alpha=eta*np.sqrt(1-0.99**(i+1))/(1-0.9**(i+1))
                W1=W1-alpha*m1/np.sqrt(G1+0.001)
                W2=W2-alpha*m2/np.sqrt(G2+0.001)
                '''
                '''
                #original
                W1=W1-eta*(g1)
                W2=W2-eta*(g2)
                bias=bias-eta*(b)
                '''
                #=================End of Calculate gradient descent=================

            #=================Calculate Loss================= 
            #z=np.dot(x,W1)+bias
            z=np.dot(x**2,W2)+np.dot(x,W1)+bias
            L=sum((y-z)**2)/2
            #=================End of Calculate Loss================= 

            #=================Print loss================= 
            if(i%50==0):
                print(i,":",L,"delta L:",(L_post-L))
                try:
                    #old.remove()
                    pass
                except Exception:
                    pass
                #plt.figure(1)
                #plt.plot(t,y, 'bo',)
                #old,=plt.plot(t,z, 'ro',)
                try:
                    #plt.pause(0.01)
                    pass

                except Exception:
                    pass
            L_post=L
            #validation
            if(i%50==0 and i>1):
                z_validation=np.dot(validation**2,W2)+np.dot(validation,W1)+bias
                L_validation=sum((validation_target-z_validation)**2)/2
                print("VALIDATION:",np.sqrt(L_validation*2/float(len(z_validation))))
            #=================End of Print loss=================

        #=================Save parameter=================
            if(i%100==0 and i>1):
                #with open('output.pickle','wb') as file:
                    #model=[W1,W2,bias]
                    #pickle.dump(model,file)
                    pass
        with open(output_pickle,'wb') as file:
            model=[W1,W2,bias]
            pickle.dump(model,file)
        print("save!")
        #=================End of Save parameter=================
        #=================End of Training iteration=================
        
    else:
        #=================Output result=================
        #input model
        with open('output.pickle','rb') as file:
            model=pickle.load(file)
            W1=(model[0])
            W2=(model[1])
            bias=model[2]
        #=================Calculate result=================
        x=np.array(test)
        z=np.dot(x**2,W2)+np.dot(x,W1)+bias
        for i in range(240):
            if(z[i]<0):
                z[i]=pm_25[i]
                pass
        #plot
        #t=np.arange(0,240,1)
        #plt.plot(t,z,'bo')
        
        result=[]
        for i in range(240):      
            result.append(z[i])
        #=================End of Calculate result=================

        #output csv 
        data=[]
        data.append(['id','value'])
        for k in range(240):
            data.append(['id_'+str(k),result[k]])
        f = open(result_csv,"w")
        w = csv.writer(f)
        w.writerows(data)
        f.close()
        print("Finish!")
        #=================End of Output result=================
    
if __name__=="__main__":
    main(eta=0.5,output_pickle='output.pickle',train_condition=False)
    
