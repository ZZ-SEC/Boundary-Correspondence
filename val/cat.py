import pandas as pd
import numpy as np
index_all=[]
for i in range(0,10):
    index=pd.read_csv("indexes_val"+str(i)+".csv")
    index_all.append(index)
index_all=np.concatenate(index_all,axis=0)
df=pd.DataFrame(index_all,columns=["a","b","c","d"])
df.to_csv("index_val.csv",index=False)