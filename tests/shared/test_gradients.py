"""Finite-difference checks for the new self-supervised loss through history."""
import json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from training.train import Model,run,ROOT,DESIGNS
episode=json.loads((ROOT/'parity-input.json').read_text())
data=[{'accounts':episode['accounts'],'events':episode['events'][:8]}]
for design in DESIGNS:
 if DESIGNS[design].get('family') in ('xgboost','temporal_family'):continue
 m=Model(design=design);loss,_=run(m,data,True);loss.backward();errors={}
 for name in ['Wz','Wn','edge0','q1','rq','pair','amount','gap']:
  if name not in m.p:continue
  p=m.p[name];idx=np.unravel_index(np.argmax(np.abs(p.g)),p.g.shape);analytic=p.g[idx];old=p.d[idx];eps=1e-5
  p.d[idx]=old+eps;plus=run(m,data,True)[0].d
  p.d[idx]=old-eps;minus=run(m,data,True)[0].d;p.d[idx]=old
  err=abs((plus-minus)/(2*eps)-analytic);errors[name]=float(err);assert err<1e-6,(name,err)
 print('PASS gradient checks',design,errors)
