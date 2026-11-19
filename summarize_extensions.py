"""Independent-fit intervals and paired comparisons for the extension experiments."""
import pathlib,csv,collections,numpy as np
from scipy.stats import t
P=pathlib.Path(__file__).resolve().parent/'results'

def ci(values):
    x=np.asarray(values,dtype=float);m=float(x.mean())
    h=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return m,m-h,m+h

def save(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)

rows=list(csv.DictReader((P/'extensions_metrics.csv').open()))
metrics=['coverage','width_uniform_age','certified_fraction','false_certificate','drop_fraction','mean_age']
out=[]
for aggregation in ['plant','balanced']:
    groups=collections.defaultdict(list)
    for row in rows:
        key=((row['plant'] if aggregation=='plant' else 'balanced'),row['law'],row['method'],row['policy'])
        groups[key].append(row)
    for key,group in groups.items():
        rr=dict(zip(['plant','law','method','policy'],key))
        for metric in metrics:
            byrep=collections.defaultdict(list)
            for row in group:
                if row.get(metric,'')!='':byrep[row['rep']].append(float(row[metric]))
            if not byrep:continue
            vals=[np.mean(x) for x in byrep.values()]
            for suffix,value in zip(['mean','ci_low','ci_high'],ci(vals)):rr[metric+'_'+suffix]=value
        out.append(rr)
save('extensions_summary.csv',out)
paired=[]
for aggregation in ['plant','balanced']:
    groups=collections.defaultdict(dict)
    for row in rows:groups[(row['plant'],row['law'],row['rep'],row['policy'])][row['method']]=row
    coll=collections.defaultdict(lambda:collections.defaultdict(list))
    for (plant,law,rep,policy),methods in groups.items():
        for comparator in ['primitive_calibrated','constant','max_shape','tail_shape','primitive','observed_mask']:
            for metric in ['coverage','width_uniform_age','certified_fraction','false_certificate']:
                if methods['all_window'].get(metric,'')=='':continue
                ours=float(methods['all_window'][metric]);other=float(methods[comparator][metric])
                value=1-ours/other if metric=='width_uniform_age' else ours-other
                key=(plant if aggregation=='plant' else 'balanced',law,policy,comparator,metric)
                coll[key][rep].append(value)
    for key,byrep in coll.items():
        vals=[np.mean(x) for x in byrep.values()]
        rr=dict(zip(['plant','law','policy','comparator','metric'],key));rr.update(zip(['mean','ci_low','ci_high'],ci(vals)));paired.append(rr)
save('extensions_paired.csv',paired)
mis=list(csv.DictReader((P/'mismatch_metrics.csv').open()));groups=collections.defaultdict(list)
for r in mis:groups[(r['law'],r['delta'])].append(r)
summary=[]
for key,group in groups.items():
    row=dict(zip(['law','delta'],key))
    for m in ['coverage','certified_fraction','false_certificate']:
        row.update((m+'_'+s,v) for s,v in zip(['mean','ci_low','ci_high'],ci([float(r[m]) for r in group])))
    summary.append(row)
save('mismatch_summary.csv',summary)
print('Created extension, paired-comparison, and mismatch summaries.')
