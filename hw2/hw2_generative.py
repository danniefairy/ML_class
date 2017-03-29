import numpy as np
import sys
import csv

def data_extraction():
    raw_train=sys.argv[1]
    raw_test=sys.argv[2]
    train= sys.argv[3]
    target=sys.argv[4]
    test=sys.argv[5]
    predict=sys.argv[6]
    #train="./data/X_train.csv"
    #target="./data/Y_train.csv"
    #test="./data/X_test.csv"
    #predict="./result/res.csv"
    
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
    
    return train_data,target_data,test_data,predict
    


def mean_cov(train_data,target_data):
    #change type to np&float
    train_data=np.array(train_data)
    t_f=train_data.astype(np.float)
    
    #split >50 and <50
    bigger=[]
    smaller=[]
    for i in range(len(t_f)):
        if(target_data[i][0]=='1'):
            bigger.append(t_f[i])
        else:
            smaller.append(t_f[i])

    #find mean
    B=np.array(bigger)
    B_T=np.transpose(B)
    B_mu=[]
    for i in range(106):
        B_mu.append(B_T[i].mean())
    
    S=np.array(smaller)
    S_T=np.transpose(S)
    S_mu=[]
    for i in range(106):
        S_mu.append(S_T[i].mean())

    #find cov
    B_cov=[]
    for i in range(len(B)):
        temp=np.array([])
        if(i==0):
            temp=np.reshape(B[0]-B_mu,(-1,1))
            B_cov=temp.dot(temp.transpose())
        else:
            temp=np.reshape(B[i]-B_mu,(-1,1))
            B_cov+=temp.dot(temp.transpose())
    B_cov=B_cov/len(B)
    
    S_cov=[]
    for i in range(len(S)):
        temp=np.array([])
        if(i==0):
            temp=np.reshape(S[0]-S_mu,(-1,1))
            S_cov=temp.dot(temp.transpose())
        else:
            temp=np.reshape(S[i]-S_mu,(-1,1))
            S_cov+=temp.dot(temp.transpose())
    S_cov=S_cov/len(S)
    
    cov= (B_cov*len(B)+S_cov*len(S))/(len(B)+len(S))

    '''
    #cov test
    a=[1,3,2]
    a=np.reshape(a,(-1,1))
    print(a)
    print(a.dot(a.transpose()))
    '''
    return B_mu,S_mu,cov,B,S
    
def main():
    #data extraction
    train_data,target_data,test_data,predict=data_extraction()
    
    #mean,cov and splitted data
    B_mu,S_mu,cov,B,S=mean_cov(train_data,target_data)
    
    #P(B) P(S)
    P_B=len(B)/(len(B)+len(S))
    P_S=len(S)/(len(B)+len(S))
    
    '''
    #validation
    score_B=0
    for i in range(len(B)):
        #Gausssian Distribution
        x=B[i]
        index_B=(-0.5)*(np.dot(np.dot(x-B_mu,np.linalg.inv(cov)),(x-B_mu)))
        f_B_x=np.exp(index_B)
        index_S=(-0.5)*(np.dot(np.dot(x-S_mu,np.linalg.inv(cov)),(x-S_mu)))
        f_S_x=np.exp(index_S)

        #Result
        P_B_x=(P_B*f_B_x)/(P_B*f_B_x+P_S*f_S_x)
        
        if(P_B_x>=0.5):
            score_B+=1
        
        #determinant and inverse matrix
        #a=[[-1,-2],[3,-4]]
        #print(np.linalg.det(a))
        #print(np.linalg.inv(a))
        
    score_S=0
    for i in range(len(S)):
        #Gausssian Distribution
        x=S[i]
        index_B=(-0.5)*(np.dot(np.dot(x-B_mu,np.linalg.inv(cov)),(x-B_mu)))
        f_B_x=np.exp(index_B)
        index_S=(-0.5)*(np.dot(np.dot(x-S_mu,np.linalg.inv(cov)),(x-S_mu)))
        f_S_x=np.exp(index_S)

        #Result
        P_B_x=(P_B*f_B_x)/(P_B*f_B_x+P_S*f_S_x)
        
        if(P_B_x<0.5):
            score_S+=1
        
        #determinant and inverse matrix
        #a=[[-1,-2],[3,-4]]
        #print(np.linalg.det(a))
        #print(np.linalg.inv(a))
        
    print((score_B+score_S)/(len(B)+len(S)))
    '''
    test_data=np.array(test_data)
    test_data=test_data.astype(np.float)
    
    #testing data
    result=[]
    for i in range(len(test_data)):
        #Gausssian Distribution
        x=test_data[i]
        index_B=(-0.5)*(np.dot(np.dot(x-B_mu,np.linalg.inv(cov)),(x-B_mu)))
        f_B_x=np.exp(index_B)
        index_S=(-0.5)*(np.dot(np.dot(x-S_mu,np.linalg.inv(cov)),(x-S_mu)))
        f_S_x=np.exp(index_S)
        #Result
        P_B_x=(P_B*f_B_x)/(P_B*f_B_x+P_S*f_S_x)
        res =1 if P_B_x>0.5 else 0
        result.append(res)

    #output csv
    data=[]
    data.append(['id','label'])
    for k in range(len(result)):
        data.append([k+1,result[k]])
    f = open(predict,"w")
    w = csv.writer(f)
    w.writerows(data)
    f.close()
    print("finish!")
if __name__=="__main__":   
    main()