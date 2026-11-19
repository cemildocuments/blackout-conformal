"""Produce numerical summaries and paired differences from experiment records."""
import csv,pathlib,collections,numpy as np
from scipy.stats import t
P=pathlib.Path(__file__).resolve().parent/'results'
def ci(x):
 x=np.asarray(x,dtype=float)
 if not np.all(np.isfinite(x)):return float(np.mean(x)),float('nan'),float('nan')
 m=float(np.mean(x));h=float(t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
 return m,m-h,m+h
for mode in ['main','sensitivity']:
 rows=list(csv.DictReader((P/f'{mode}_metrics.csv').open()));groups=collections.defaultdict(list)
 keycols=['plant','law','T','L','ntrain','ncal','ntest','method','policy']
 for row in rows:groups[tuple(row[k] for k in keycols)].append(row)
 summary=[]
 for key,group in groups.items():
  out=dict(zip(keycols,key));out['repetitions']=len(group)
  for metric in ['coverage','width','width_uniform_age','false_certificate','certified_fraction','safe_certified_fraction','true_safe_fraction']:
   xx=[float(a[metric]) for a in group if a.get(metric,'')!='']
   if not xx:continue
   for suffix,value in zip(['mean','ci_low','ci_high'],ci(xx)):out[metric+'_'+suffix]=value
  summary.append(out)
 cols=list(dict.fromkeys(k for row in summary for k in row))
 with (P/f'{mode}_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(summary)
 # Paired comparisons preserve identical training, calibration, and test noise.
 cells=collections.defaultdict(dict)
 for row in rows:cells[tuple(row[k] for k in keycols if k!='method')+(row['rep'],)][row['method']]=row
 paired=collections.defaultdict(list)
 for key,cell in cells.items():
  for alternative in ['observed_mask','age_union','primitive','constant','gaussian_union']:
   for metric in ['coverage','width_uniform_age']:
    ours=float(cell['all_window'][metric]);other=float(cell[alternative][metric])
    val=ours-other if metric=='coverage' else 1-ours/other
    paired[(key[0],key[1],key[3],key[5],key[7],alternative,metric)].append(val)
 with (P/f'{mode}_paired.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['plant','law','L','ncal','policy','comparator','metric','mean','ci_low','ci_high'])
  for key,val in paired.items():w.writerow(list(key)+list(ci(val)))
print('Created summary and paired-comparison CSV files.')
