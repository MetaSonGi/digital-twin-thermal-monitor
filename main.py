"""Fit an interpretable thermal baseline; score held-out residuals."""
import argparse, csv, json, math, statistics
from pathlib import Path

def fit(rows):
    if len(rows)<4: raise ValueError('at least 4 training samples required')
    x=[r['load'] for r in rows]; y=[r['temperature']-r['ambient'] for r in rows]
    xm=statistics.mean(x); ym=statistics.mean(y)
    variance=sum((v-xm)**2 for v in x)
    if variance==0: raise ValueError('training load must vary')
    slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/variance
    intercept=ym-slope*xm
    residuals=[b-(intercept+slope*a) for a,b in zip(x,y)]
    scale=max(0.1,math.sqrt(sum(v*v for v in residuals)/(len(rows)-2)))
    return {'intercept':intercept,'load_coefficient':slope,'residual_scale':scale,
            'load_min':min(x),'load_max':max(x)}

def score(model,row,threshold=4.0):
    expected=row['ambient']+model['intercept']+model['load_coefficient']*row['load']
    residual=row['temperature']-expected
    outside=not model['load_min']<=row['load']<=model['load_max']
    z=abs(residual)/model['residual_scale']
    return {**row,'expected_temperature':round(expected,3),'residual':round(residual,3),
            'score':round(z,3),'status':'out_of_domain' if outside else 'anomaly' if z>threshold else 'normal'}

def run(path,train_count=12,threshold=4.0):
    if threshold<=0: raise ValueError('threshold must be positive')
    with open(path,newline='',encoding='utf-8') as f:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    if not 4<=train_count<len(rows): raise ValueError('training split must leave held-out data')
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('load','ambient','temperature')): raise ValueError('non-finite sensor data')
    model=fit(rows[:train_count])
    return {'model':model,'training_samples':train_count,'results':[score(model,r,threshold) for r in rows[train_count:]]}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input',default=str(Path(__file__).with_name('sample.csv')))
    p.add_argument('--train-count',type=int,default=12); p.add_argument('--threshold',type=float,default=4)
    a=p.parse_args(); print(json.dumps(run(a.input,a.train_count,a.threshold),ensure_ascii=False,indent=2))
