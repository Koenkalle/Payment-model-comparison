"""Shared differentiable GRU/graph primitives and likelihood training loop."""
from ._common import *
from training.autodiff import T,cat,gather
def ce(logits,targets,mask):
    a=logits.d;prob=np.exp(a-a.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    ix=np.arange(len(targets));m=mask/max(1,mask.sum())
    out=T(float((-np.log(np.maximum(prob[ix,targets],1e-30))*m).sum()),(logits,))
    def back():
        g=prob.copy();g[ix,targets]-=1;logits.acc(out.g*g*m[:,None])
    out.back=back;return out

class NeuralModel:
    def __init__(self,seed=512,design='gru_attention'):
        self.design=self.architecture;self.hidden=self.design['hidden'];self.layers=self.design['layers']
        H=self.hidden;rng=np.random.default_rng(seed);self.p={}
        def weight(name,ins,outs):self.p[name]=T(rng.normal(0,.7/math.sqrt(ins),(ins,outs)))
        def bias(name,n):self.p[name]=T(np.zeros(n))
        if self.design['memory']=='gru':
            for gate in ['z','r','n']:weight('W'+gate,2*H+6,H);bias('b'+gate,H)
        for l in range(self.layers):
            for m in (['q','k','v','self']if self.design['readout']=='attention'else ['v','self']):weight(f'{m}{l}',H,H)
            weight(f'edge{l}',4,H);bias(f'out{l}',H)
        if H:weight('rq',H,H);weight('rk',H,H)
        weight('pair',3,1)
        for name,size in [('amount',len(AMOUNT)+1),('gap',len(GAP)+2)]:weight(name,2*H+5,size);bias(name+'b',size)
    def update(self,h,event):
        if self.design['memory']=='none':return h
        mask,ids,f=event;p=self.p;cp=gather(h,ids)*(1-f[...,2:3]);x=cat(cp,T(f));joined=cat(x,h)
        z=(joined@p['Wz']+p['bz']).sigmoid();r=(joined@p['Wr']+p['br']).sigmoid()
        nxt=(cat(x,r*h)@p['Wn']+p['bn']).tanh()
        return ((1+z*-1)*h+z*nxt)*mask+h*(1-mask)
    def readout(self,h,graph):
        ids,attrs,valid=graph;p=self.p;H=self.hidden
        for l in range(self.layers):
            nbr=gather(h,ids);edge=T(attrs)@p[f'edge{l}'];val=nbr@p[f'v{l}']+edge
            if self.design['readout']=='attention':
                key=nbr@p[f'k{l}']+edge;q=(h@p[f'q{l}']).expand(2);w=((q*key).sum(-1)*H**-.5+(1-valid)*-1e9).softmax()
            else:w=T(valid/valid.sum(-1,keepdims=True))
            h=(h@p[f'self{l}']+(w.expand(-1)*val).sum(2)+p[f'out{l}']).tanh()
        return h
    def predict(self,z,query):
        u,v,pairs,context=query;p=self.p;sender=gather(z,u);receiver=gather(z,v)
        r=((sender@p['rq']).expand(1)*(z@p['rk'])).sum(-1)*self.hidden**-.5 if self.hidden else T(np.zeros(z.d.shape[:2]))
        r=r+(T(pairs)@p['pair']).sum(-1)+T(np.eye(z.d.shape[1])[u])*-1e9
        x=cat(sender,receiver,T(context))
        return [r,x@p['amount']+p['amountb'],x@p['gap']+p['gapb']]

def prepare(episodes):
    B=len(episodes);N=episodes[0]['accounts'];last=np.zeros((B,N));seen=np.zeros((B,N));now=np.zeros(B)
    oc=np.zeros((B,N));ic=oc.copy();ov=oc.copy();iv=oc.copy();pairs=np.zeros((B,N,N));lists=[[[]for _ in range(N)]for _ in range(B)];out=[]
    for step in range(len(episodes[0]['events'])):
        ids=np.tile(np.arange(N)[None,:,None],(B,1,K));attrs=np.zeros((B,N,K,4));valid=np.zeros((B,N,K));valid[:,:,0]=1
        mask=np.zeros((B,N,1));cp=np.zeros((B,N),int);f=np.zeros((B,N,6));us=np.zeros(B,int);vs=us.copy()
        pf=np.zeros((B,N,3));ctx=np.zeros((B,5));targets=np.zeros((3,B),int);lossmask=np.zeros(B)
        for b,ep in enumerate(episodes):
            e=ep['events'][step];u,v,t=e['u'],e['v'],e['t'];us[b]=max(0,u);vs[b]=v
            # Read at the preceding observed time; current time and amount
            # are targets, not inputs to their own prediction.
            for n in range(N):
                for k,(other,when,amt,role)in enumerate(lists[b][n][-K+1:][::-1],1):
                    ids[b,n,k]=other;attrs[b,n,k]=[math.log1p(amt)/8,math.log1p(max(0,now[b]-when))/6,role,1];valid[b,n,k]=1
            if u>=0:
                lossmask[b]=1;pf[b,:,0]=np.log1p(pairs[b,u])/3;pf[b,:,1]=np.log1p(ic[b])/6;pf[b,:,2]=(pairs[b,u]+pairs[b,:,u]>0)
                ctx[b]=[math.log1p(ov[b,u]/max(1,oc[b,u]))/8,math.log1p(iv[b,v]/max(1,ic[b,v]))/8,math.log1p(seen[b,u])/6,math.log1p(seen[b,v])/6,math.log1p(pairs[b,u,v])/3]
                targets[:,b]=[v,np.searchsorted(AMOUNT,e['amount'],side='right'),np.searchsorted(GAP,max(0,t-last[b,u]),side='right')if seen[b,u]else len(GAP)+1]
            actors=[(v,u,2 if u<0 else 1)]+([(u,v,0)]if u>=0 else [])
            for n,other,role in actors:
                mask[b,n]=1;cp[b,n]=max(0,other);f[b,n,role]=1;f[b,n,3]=float(e.get('settled',True)is False)
                f[b,n,4:]=[math.log1p(e['amount'])/8,math.log1p(max(0,t-last[b,n]))/6];last[b,n]=t;seen[b,n]+=1
            if u>=0 and e.get('settled',True):
                lists[b][u].append((v,t,e['amount'],-1));lists[b][v].append((u,t,e['amount'],1));oc[b,u]+=1;ic[b,v]+=1
                ov[b,u]+=e['amount'];iv[b,v]+=e['amount'];pairs[b,u,v]+=1
            now[b]=t
        out.append(((mask,cp,f),(ids,attrs,valid),(us,vs,pf,ctx),targets,lossmask))
    return out

def run(model,episodes,train=False,trace=False):
    h=T(np.zeros((len(episodes),episodes[0]['accounts'],model.hidden)));losses=[];records=[]
    for event,graph,q,targets,mask in prepare(episodes):
        z=model.readout(h,graph);pred=model.predict(z,q)
        if train and mask.sum():losses.append(sum([ce(a,b,mask)for a,b in zip(pred,targets)],T(0)))
        if trace:
            probs=[x.softmax().d for x in pred];ix=np.arange(len(episodes));scores=sum(-np.log(np.maximum(p[ix,target],1e-30))for p,target in zip(probs,targets))/math.log(2)
            records.append({'scores':scores.tolist(),'probabilities':[p.tolist()for p in probs],'memory_before':h.d.tolist(),'embedding':z.d.tolist(),'queries':[{'u':int(q[0][b]),'v':int(q[1][b]),'pair_features':q[2][b].tolist(),'context':q[3][b].tolist()}for b in range(len(episodes))]})
        h=model.update(h,event)
        if trace:records[-1]['memory_after']=h.d.tolist()
    return sum(losses,T(0))*(1/max(1,len(losses))),records

def train_design(a,data,design,labeled=None):
    model=Model(seed=a.model_seed,design=design);rng=np.random.default_rng(7131)
    moments={n:(np.zeros_like(p.d),np.zeros_like(p.d))for n,p in model.p.items()};start=time.time();losslog=[]
    for iteration in range(1,a.steps+1):
        eps=[data[i]for i in rng.choice(len(data)-48,a.batch,replace=False)]
        for p in model.p.values():p.g.fill(0)
        loss,_=run(model,eps,True);loss.backward();losslog.append(float(loss.d));norm=math.sqrt(sum(float((p.g*p.g).sum())for p in model.p.values()))
        for name,p in model.p.items():
            g=p.g*min(1,3/max(norm,1e-12));m,v=moments[name];m[:]=.9*m+.1*g;v[:]=.999*v+.001*g*g
            p.d-=.009*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
        if iteration%20==0:print(f'{design} {iteration}/{a.steps} loss {float(loss.d):.4f}; {time.time()-start:.0f}s',flush=True)
    val,_=run(model,data[-24:],True)
    summary={'method':'Self-supervised recipient + amount-bin + gap-bin likelihood','fraud_labels_used':0,'default_warmup_requests':int(a.warmup),'model_seed':a.model_seed,'sampling_seed':7131,'generator_seed':1731,'training_data_sha256':hashlib.sha256((ROOT/'training-data.json').read_bytes()).hexdigest(),'training_pool_episodes':len(data)-48,'steps':a.steps,'batch':a.batch,'events_per_episode':48,'accounts_per_episode':24,'learning_rate':.009,'last_20_steps_mean_nll':float(np.mean(losslog[-20:])),'heldout_nll_nats_per_request':float(val.d),'notice':'Synthetic predictive loss; one model initialization; no fraud labels or reports enter training.'}
    weights={n:p.d.tolist()for n,p in model.p.items()};supervised=fit_supervised_head(model,labeled) if labeled is not None else None
    payload={'version':3,'id':design,'adapter':'categorical_graph','label':DESIGNS[design]['label'],'architecture':DESIGNS[design],'hidden':model.hidden,'neighbors':K,'amount_bins':AMOUNT,'gap_bins':GAP,'weights':weights,'training':summary,'supervised':supervised,'policy':{'default_warmup_requests':int(a.warmup)},'parameters':sum(p.d.size for p in model.p.values()),'checkpoint_id':hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest()[:16]}
    (ROOT/'models'/f'{design}.json').write_text(json.dumps(payload,separators=(',',':')))
    episode=data[-1];_,trace=run(model,[episode],trace=True)
    if supervised is not None:(ROOT/'parity-supervised').mkdir(exist_ok=True);(ROOT/'parity-supervised'/f'{design}.json').write_text(json.dumps({'episode':episode,'expected':supervised_trace(model,supervised,episode)},separators=(',',':')))
    (ROOT/'parity'/f'{design}.json').write_text(json.dumps({'episode':episode,'expected':trace},separators=(',',':')))
    if design==CATALOG['default']:
        (ROOT/'model.json').write_text(json.dumps(payload,separators=(',',':')));(ROOT/'training-summary.json').write_text(json.dumps(summary,indent=2))
        (ROOT/'parity-input.json').write_text(json.dumps(episode));(ROOT/'parity-expected.json').write_text(json.dumps(trace))
    print(design,'held-out predictive NLL',float(val.d),'parameters',payload['parameters'],flush=True)

def supervised_row(record):
    query=record['queries'][0];z=record['embedding'][0];return np.asarray(z[query['u']]+z[query['v']]+query['context']+[record['scores'][0]],dtype=float)

def supervised_rows(model,dataset):
    rows=[];labels=[];episodes=[]
    for episode_index,episode in enumerate(dataset['episodes']):
        _,trace=run(model,[episode],trace=True)
        for index,e in enumerate(episode['events']):
            if e['kind']=='payment' and int(e.get('label',-1))>=0:
                rows.append(supervised_row(trace[index]));labels.append(int(e['label']));episodes.append(episode_index)
    return np.asarray(rows,dtype=float),np.asarray(labels,dtype=float),np.asarray(episodes,dtype=int)

def fit_supervised_head(model,dataset,steps=220,learning_rate=.08):
    X,y,episode_ids=supervised_rows(model,dataset);split=max(1,int(len(dataset['episodes'])*.8));train=episode_ids<split;valid=~train
    if train.sum()==0 or y[train].sum()==0:raise ValueError('Supervised mode requires at least one flagged fraud example.')
    mean=X[train].mean(0);scale=X[train].std(0);scale=np.where(scale<1e-8,1.0,scale);Xs=(X[train]-mean)/scale;yt=y[train];positive=max(1.0,float(yt.sum()));negative=max(1.0,float(len(yt)-yt.sum()));pos_weight=min(10.0,negative/positive)
    weights=np.where(yt>0,pos_weight,1.0);rate=float(np.clip(yt.mean(),1e-4,1-1e-4));w=np.zeros(Xs.shape[1]);b=math.log(rate/(1-rate))
    for _ in range(int(steps)):
        p=1/(1+np.exp(-np.clip(Xs@w+b,-50,50)));error=(p-yt)*weights;w-=learning_rate*(Xs.T@error/max(1,len(yt)));b-=learning_rate*float(error.mean())
    raw=(X[valid]-mean)/scale@w+b;vp=1/(1+np.exp(-np.clip(raw,-50,50)));vy=y[valid];val_loss=float(-(vy*np.log(np.maximum(vp,1e-12))+(1-vy)*np.log(np.maximum(1-vp,1e-12))).mean())
    head={'type':'logistic_fraud_head','mean':mean.tolist(),'scale':scale.tolist(),'weights':w.tolist(),'bias':float(b),'training':{'fraud_labels_used':int(yt.sum()),'fraud_flags_requested':int(dataset.get('flags_requested',0)),'validation_flags_used':int(dataset.get('validation_flags_used',0)),'training_rows':int(train.sum()),'validation_rows':int(valid.sum()),'steps':int(steps),'learning_rate':learning_rate,'scale_pos_weight':pos_weight,'heldout_logloss':val_loss,'notice':'Supervised head trained offline from flagged historical outcomes; scores are not calibrated production fraud probabilities.'}}
    return head

def supervised_probability(head,row):
    x=(np.asarray(row,dtype=float)-np.asarray(head['mean']))/np.asarray(head['scale']);return float(_sigmoid(float(np.dot(x,np.asarray(head['weights']))+head['bias'])))

def supervised_trace(model,head,episode):
    _,trace=run(model,[episode],trace=True);out=[]
    for event,record in zip(episode['events'],trace):
        if event['kind']=='payment':
            p=supervised_probability(head,supervised_row(record));score=-math.log2(max(1e-12,1-p));probabilities=[[[1-p,p]],[[1.0]],[[1.0]]]
        else:score=0.0;probabilities=[[[1.0]],[[1.0]],[[1.0]]]
        out.append({**record,'scores':[score],'probabilities':probabilities})
    return out


def Model(seed=512,design='gru_attention'):
    from framework.registry import model_entry,load_module
    return load_module(model_entry(design)['python_module']).Model(seed=seed,design=design)
