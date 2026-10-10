import json,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,precision_score,recall_score,confusion_matrix

def dataset(n=6000,seed=42):
    rng=np.random.default_rng(seed)
    amount=rng.lognormal(3.8,1.2,n);velocity=rng.poisson(2,n);distance=rng.exponential(40,n)
    x=np.column_stack([np.log1p(amount),velocity,np.log1p(distance)])
    probability=1/(1+np.exp(-(-9+0.65*x[:,0]+.65*x[:,1]+.5*x[:,2])))
    y=(rng.random(n)<probability).astype(int)
    return x,y

def choose_threshold(y,p,min_precision=.35):
    candidates=[]
    for t in np.linspace(.05,.95,91):
        predicted=p>=t
        precision=precision_score(y,predicted,zero_division=0);recall=recall_score(y,predicted,zero_division=0)
        if precision>=min_precision:candidates.append((recall,float(t)))
    return max(candidates,key=lambda v:(v[0],-v[1]))[1] if candidates else 1.0

def experiment():
    x,y=dataset();train=3600;valid=4800
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
    model.fit(x[:train],y[:train]);threshold=choose_threshold(y[train:valid],model.predict_proba(x[train:valid])[:,1])
    p=model.predict_proba(x[valid:])[:,1];pred=p>=threshold
    return {'data':'synthetic; sequential train/validation/test split','train_n':train,'validation_n':valid-train,'test_n':len(y)-valid,'test_prevalence':float(y[valid:].mean()),'threshold_from_validation':threshold,'average_precision':float(average_precision_score(y[valid:],p)),'precision':float(precision_score(y[valid:],pred,zero_division=0)),'recall':float(recall_score(y[valid:],pred,zero_division=0)),'confusion_matrix':confusion_matrix(y[valid:],pred).tolist(),'coefficients_standardized':model[-1].coef_[0].tolist()}
if __name__=='__main__': print(json.dumps(experiment(),indent=2))
