"""Offline trainers for the self-supervised temporal models and XGBoost.

The temporal models learn recipient, amount-bin, and sender-gap-bin likelihoods
without fraud labels.  The XGBoost entry is a small NumPy implementation of
the regularized, second-order binary logistic tree booster; it exports trees so
the browser demo has no Python runtime dependency.
"""
import argparse,json,math,subprocess,time,hashlib
from pathlib import Path
import numpy as np
from autodiff import T,cat,gather
ROOT=Path(__file__).resolve().parent
H,K=8,5
AMOUNT=[15,35,75,150,300,650,1500,4000,10000]
GAP=[1,5,20,60,180,720,2880]
XGB_FEATURES=[
    'log_amount','amount_bin','sender_out_count','sender_in_count',
    'recipient_out_count','recipient_in_count','sender_out_value',
    'sender_in_value','recipient_out_value','recipient_in_value',
    'sender_seen','recipient_seen','sender_gap','recipient_gap',
    'pair_out_count','pair_total_count','prior_contact',
    'sender_recent_in_count','sender_recent_out_count',
    'recipient_recent_in_count','recipient_recent_out_count',
    'sender_recent_in_value','sender_recent_out_value',
    'recipient_recent_in_value','recipient_recent_out_value',
    'amount_vs_sender_out_mean','amount_vs_recipient_in_mean'
]
CATALOG=json.loads((ROOT/'designs.json').read_text())
DESIGNS={d['id']:d for d in CATALOG['designs']}
SEQ_VARIANTS={
    'dygformer':{'use_pair':False,'use_gnn':False,'recipient_pair_weight':.4,'recipient_activity_weight':.2,'recipient_temporal_weight':1.2,'recipient_pair_state_weight':0.0,'recipient_graph_weight':0.0},
    'tami':{'use_pair':True,'use_gnn':False,'recipient_pair_weight':1.2,'recipient_activity_weight':.2,'recipient_temporal_weight':.25,'recipient_pair_state_weight':1.0,'recipient_graph_weight':0.0},
    'dyg_tami':{'use_pair':True,'use_gnn':False,'recipient_pair_weight':.8,'recipient_activity_weight':.2,'recipient_temporal_weight':.8,'recipient_pair_state_weight':1.0,'recipient_graph_weight':0.0},
    'dyg_tami_gnn':{'use_pair':True,'use_gnn':True,'recipient_pair_weight':.65,'recipient_activity_weight':.15,'recipient_temporal_weight':.6,'recipient_pair_state_weight':.8,'recipient_graph_weight':.8}
}

def ce(logits,targets,mask):
    a=logits.d;prob=np.exp(a-a.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    ix=np.arange(len(targets));m=mask/max(1,mask.sum())
    out=T(float((-np.log(np.maximum(prob[ix,targets],1e-30))*m).sum()),(logits,))
    def back():
        g=prob.copy();g[ix,targets]-=1;logits.acc(out.g*g*m[:,None])
    out.back=back;return out

class Model:
    def __init__(self,seed=512,design='gru_attention'):
        self.design=DESIGNS[design];self.hidden=self.design['hidden'];self.layers=self.design['layers']
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

def _log1p(x,scale=1.0):
    return math.log1p(max(0.0,float(x)))/scale

def xgb_state(n):
    return {'last':np.zeros(n),'seen':np.zeros(n),'oc':np.zeros(n),'ic':np.zeros(n),
            'ov':np.zeros(n),'iv':np.zeros(n),'pairs':np.zeros((n,n)),
            'incidents':[[] for _ in range(n)],'now':0.0}

def xgb_feature_row(state,e):
    u,v,t,amount=int(e['u']),int(e['v']),float(e['t']),float(e['amount'])
    assert u>=0 and v>=0
    def recent(n,role,window=60.0):
        events=[x for x in state['incidents'][n] if t-x[1]<=window and x[3]==role]
        return len(events),sum(x[2] for x in events)
    si,so=recent(u,1),recent(u,-1);ri,ro=recent(v,1),recent(v,-1)
    sender_mean=state['ov'][u]/max(1.0,state['oc'][u]);recipient_mean=state['iv'][v]/max(1.0,state['ic'][v])
    gap_u=max(0.0,t-state['last'][u]) if state['seen'][u] else 0.0
    gap_v=max(0.0,t-state['last'][v]) if state['seen'][v] else 0.0
    return np.array([
        _log1p(amount,8),np.searchsorted(AMOUNT,amount,side='right')/len(AMOUNT),
        _log1p(state['oc'][u],6),_log1p(state['ic'][u],6),
        _log1p(state['oc'][v],6),_log1p(state['ic'][v],6),
        _log1p(state['ov'][u],10),_log1p(state['iv'][u],10),
        _log1p(state['ov'][v],10),_log1p(state['iv'][v],10),
        _log1p(state['seen'][u],6),_log1p(state['seen'][v],6),
        _log1p(gap_u,6),_log1p(gap_v,6),
        _log1p(state['pairs'][u,v],3),_log1p(state['pairs'][u,v]+state['pairs'][v,u],3),
        float(state['pairs'][u,v]+state['pairs'][v,u]>0),
        _log1p(si[0],4),_log1p(so[0],4),_log1p(ri[0],4),_log1p(ro[0],4),
        _log1p(si[1],10),_log1p(so[1],10),_log1p(ri[1],10),_log1p(ro[1],10),
        _log1p(amount/max(1.0,sender_mean),8),_log1p(amount/max(1.0,recipient_mean),8)
    ],dtype=float)

def xgb_apply(state,e):
    t=float(e['t']);u,v=int(e['u']),int(e['v']);settled=e.get('settled',True) is not False
    for n in ([v,u] if u>=0 else [v]):
        state['last'][n]=t;state['seen'][n]+=1
    if e['kind']=='payment' and u>=0 and settled:
        amount=float(e['amount']);state['incidents'][u].append((v,t,amount,-1));state['incidents'][v].append((u,t,amount,1))
        state['oc'][u]+=1;state['ic'][v]+=1;state['ov'][u]+=amount;state['iv'][v]+=amount;state['pairs'][u,v]+=1
    state['now']=t

def xgb_rows(dataset):
    rows=[];labels=[];episodes=[]
    for episode_index,episode in enumerate(dataset['episodes']):
        state=xgb_state(episode['accounts'])
        for e in episode['events']:
            if e['kind']=='payment' and int(e.get('label',-1))>=0:
                rows.append(xgb_feature_row(state,e));labels.append(int(e['label']));episodes.append(episode_index)
            xgb_apply(state,e)
    return np.asarray(rows,dtype=float),np.asarray(labels,dtype=float),np.asarray(episodes,dtype=int)

def _sigmoid(x):
    return 1.0/(1.0+math.exp(-max(-50.0,min(50.0,float(x)))))

def _tree_predict(tree,row):
    node=tree
    while 'leaf' not in node:
        node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
    return float(node['leaf'])

def _grow_tree(X,g,h,indices,depth,max_depth,min_child,reg_lambda,gamma):
    G=float(g[indices].sum());H=float(h[indices].sum())
    node={'leaf':-G/(H+reg_lambda)}
    if depth>=max_depth or len(indices)<2*min_child:return node
    parent=G*G/(H+reg_lambda);best=None
    for feature in range(X.shape[1]):
        order=indices[np.argsort(X[indices,feature],kind='mergesort')];values=X[order,feature]
        gs=np.cumsum(g[order]);hs=np.cumsum(h[order]);total_g=float(gs[-1]);total_h=float(hs[-1])
        for cut in range(min_child,len(order)-min_child+1):
            if cut<len(order) and values[cut-1]==values[cut]:continue
            left_g=float(gs[cut-1]);left_h=float(hs[cut-1]);right_g=total_g-left_g;right_h=total_h-left_h
            if left_h<=0 or right_h<=0:continue
            gain=.5*(left_g*left_g/(left_h+reg_lambda)+right_g*right_g/(right_h+reg_lambda)-parent)-gamma
            if best is None or gain>best[0]:
                threshold=float(values[cut-1] if cut==len(order) else (values[cut-1]+values[cut])/2)
                best=(gain,feature,threshold)
    if best is None or best[0]<=0:return node
    _,feature,threshold=best;left=indices[X[indices,feature]<=threshold];right=indices[X[indices,feature]>threshold]
    if len(left)<min_child or len(right)<min_child:return node
    return {'feature':feature,'threshold':threshold,
            'left':_grow_tree(X,g,h,left,depth+1,max_depth,min_child,reg_lambda,gamma),
            'right':_grow_tree(X,g,h,right,depth+1,max_depth,min_child,reg_lambda,gamma)}

def xgb_predict_raw(model,X):
    raw=np.full(len(X),float(model['base_score']))
    for tree in model['trees']:
        raw+=float(model['learning_rate'])*np.asarray([_tree_predict(tree,row) for row in X])
    return raw

def fit_xgb(X,y,trees=32,depth=3,learning_rate=.08,min_child=12,reg_lambda=1.0,gamma=.0,seed=512):
    if len(X)==0 or y.sum()==0:raise ValueError('XGBoost training requires at least one flagged fraud example.')
    rng=np.random.default_rng(seed);order=rng.permutation(len(X));X=X[order];y=y[order]
    positive=max(1.0,float(y.sum()));negative=max(1.0,float(len(y)-y.sum()));scale=min(10.0,negative/positive)
    weights=np.where(y>0,scale,1.0);rate=float(np.clip(y.mean(),1e-4,1-1e-4));base=math.log(rate/(1-rate));raw=np.full(len(X),base);out=[]
    for _ in range(int(trees)):
        p=1/(1+np.exp(-np.clip(raw,-50,50)));g=(p-y)*weights;h=np.maximum(p*(1-p),1e-3)*weights
        tree=_grow_tree(X,g,h,np.arange(len(X)),0,int(depth),int(min_child),float(reg_lambda),float(gamma));out.append(tree)
        raw+=float(learning_rate)*np.asarray([_tree_predict(tree,row) for row in X])
    return {'base_score':base,'learning_rate':float(learning_rate),'trees':out,'scale_pos_weight':scale}

def xgb_trace(model,episode):
    state=xgb_state(episode['accounts']);records=[]
    for e in episode['events']:
        before=[list(x) for x in state_memory(state)]
        if e['kind']=='payment':
            row=xgb_feature_row(state,e);p=_sigmoid(xgb_predict_raw(model,row[None,:])[0]);score=-math.log2(max(1e-12,1-p))
            probs=[[[1-p,p]],[[1.0]],[[1.0]]];records.append({'scores':[score],'probabilities':probs,'memory_before':[before],'embedding':[[[] for _ in range(episode['accounts'])]][0]})
        else:records.append({'scores':[0.0],'probabilities':[[[1.0]],[[1.0]],[[1.0]]],'memory_before':[before],'embedding':[[[] for _ in range(episode['accounts'])]][0]})
        xgb_apply(state,e);records[-1]['memory_after']=[list(x) for x in state_memory(state)]
    return records

def xgb_unsupervised_score(row):
    """Causal no-label rarity score mirrored by model-adapters.js."""
    return max(0.0,.35*row[0]+.4*row[1]+1.2*(1-row[16])+.7*row[12]+.7*row[13]+.6*row[17]+.8*row[18]+.6*row[19]+.8*row[20]+.8*row[25]+.8*row[26])

def xgb_unsupervised_trace(episode):
    state=xgb_state(episode['accounts']);records=[]
    for e in episode['events']:
        before=[list(x) for x in state_memory(state)]
        if e['kind']=='payment':
            row=xgb_feature_row(state,e);score=xgb_unsupervised_score(row)
            probabilities=[[[1.0]],[[1.0]],[[1.0]]]
            records.append({'scores':[score],'probabilities':probabilities,'memory_before':[before],'embedding':[[[] for _ in range(episode['accounts'])]][0]})
        else:records.append({'scores':[0.0],'probabilities':[[[1.0]],[[1.0]],[[1.0]]],'memory_before':[before],'embedding':[[[] for _ in range(episode['accounts'])]][0]})
        xgb_apply(state,e);records[-1]['memory_after']=[list(x) for x in state_memory(state)]
    return records

def sequence_config(model):
    return model.get('sequence',{})

def sequence_history_vector(model,state,n,t):
    cfg=sequence_config(model);dim=int(cfg.get('dim',8));weights=np.asarray(cfg.get('history_weights',np.zeros((dim,8))),dtype=float);out=np.zeros(dim);total=0.0
    events=list(state['incidents'][n])[-int(cfg.get('history_limit',8)):][::-1]
    for other,when,amount,role in events:
        age=max(0.0,float(t)-float(when));weight=math.exp(-age/float(cfg.get('history_decay',720)));basis=np.asarray([1.0,float(role),math.log1p(amount)/8.0,min(age/720.0,4.0),math.sin(age/60.0),math.cos(age/60.0),math.sin(age/1440.0),math.cos(age/1440.0)])
        out+=weight*(weights@basis);total+=weight
    return np.tanh(out/total) if total else out

def sequence_pair_events(model,state,u,v):
    cfg=sequence_config(model);limit=int(cfg.get('pair_limit',6));return [e for e in state['incidents'][u] if e[0]==v and e[3]==-1][-limit:][::-1]

def sequence_pair_vector(model,state,u,v,t):
    cfg=sequence_config(model);dim=int(cfg.get('pair_dim',4));weights=np.asarray(cfg.get('pair_weights',np.zeros((dim,5))),dtype=float);out=np.zeros(dim);total=0.0
    for other,when,amount,role in sequence_pair_events(model,state,u,v):
        age=max(0.0,float(t)-float(when));weight=math.exp(-age/float(cfg.get('pair_decay',1440)));basis=np.asarray([1.0,math.log1p(amount)/8.0,min(age/720.0,4.0),math.sin(age/60.0),math.cos(age/60.0)])
        out+=weight*(weights@basis);total+=weight
    return np.tanh(out/total) if total else out

def sequence_neighbors(model,state,n):
    limit=int(sequence_config(model).get('neighbor_limit',4));seen=set();out=[]
    for event in list(state['incidents'][n])[-limit:][::-1]:
        if event[0] not in seen:seen.add(event[0]);out.append(event)
    return out

def sequence_graph_vector(model,state,n,t,depth,cache):
    cfg=sequence_config(model);key=(n,depth)
    if key in cache:return cache[key]
    base=sequence_history_vector(model,state,n,t)
    if not cfg.get('use_gnn') or depth<=0:cache[key]=base;return base
    dim=int(cfg.get('dim',8));weights=np.asarray(cfg.get('gnn_weights',np.zeros((dim,dim+4))),dtype=float);out=base.copy();total=0.0
    for other,when,amount,role in sequence_neighbors(model,state,n):
        child=sequence_graph_vector(model,state,other,t,depth-1,cache);age=max(0.0,float(t)-float(when));edge=np.asarray([float(role),math.log1p(amount)/8.0,min(age/720.0,4.0),1.0]);value=np.tanh(weights@np.concatenate([child,edge]));weight=math.exp(-age/float(cfg.get('history_decay',720)));out+=weight*value;total+=weight
    cache[key]=np.tanh(base+(out-base)/total) if total else base
    return cache[key]

def sequence_account_embedding(model,state,n,t,cache):
    cfg=sequence_config(model);return sequence_graph_vector(model,state,n,t,int(cfg.get('gnn_layers',2)),cache) if cfg.get('use_gnn') else sequence_history_vector(model,state,n,t)

def sequence_amount_logits(model,state,u,v,t):
    cfg=sequence_config(model);edges=model['amount_bins'];hist=np.zeros(len(edges)+1);decay=float(cfg.get('history_decay',720))
    for other,when,amount,role in state['incidents'][u]:
        if role==-1:hist[int(np.searchsorted(edges,amount,side='right'))]+=math.exp(-max(0.0,t-when)/decay)
    if cfg.get('use_pair'):
        for other,when,amount,role in sequence_pair_events(model,state,u,v):hist[int(np.searchsorted(edges,amount,side='right'))]+=float(cfg.get('pair_amount_weight',1.0))*math.exp(-max(0.0,t-when)/float(cfg.get('pair_decay',1440)))
    return np.log(hist+float(cfg.get('histogram_smoothing',.5)))

def sequence_gap_logits(model,state,u,v,t):
    cfg=sequence_config(model);edges=model['gap_bins'];hist=np.zeros(len(edges)+2);out=sorted([e for e in state['incidents'][u] if e[3]==-1],key=lambda e:e[1]);last=None
    for other,when,amount,role in out:
        if last is not None:hist[int(np.searchsorted(edges,max(0.0,when-last),side='right'))]+=math.exp(-max(0.0,t-when)/float(cfg.get('history_decay',720)))
        last=when
    if cfg.get('use_pair'):
        pair=sorted(sequence_pair_events(model,state,u,v),key=lambda e:e[1]);last=None
        for other,when,amount,role in pair:
            if last is not None:hist[int(np.searchsorted(edges,max(0.0,when-last),side='right'))]+=float(cfg.get('pair_gap_weight',1.0))*math.exp(-max(0.0,t-when)/float(cfg.get('pair_decay',1440)))
            last=when
    return np.log(hist+float(cfg.get('histogram_smoothing',.5)))

def sequence_softmax(values):
    values=np.asarray(values,dtype=float);shift=values-values.max();exp=np.exp(shift);return exp/exp.sum()

def sequence_prediction(model,state,e):
    cfg=sequence_config(model);n=len(state['last']);u=int(e['u']);v=int(e['v']);t=float(e['t']);cache={};sender=sequence_account_embedding(model,state,u,t,cache);history_sender=sequence_history_vector(model,state,u,t);dim=max(1,len(sender));recipient=np.full(n,-1e9)
    for candidate in range(n):
        if candidate==u:continue
        candidate_embedding=sequence_account_embedding(model,state,candidate,t,cache);history_candidate=sequence_history_vector(model,state,candidate,t);pair=sequence_pair_vector(model,state,u,candidate,t) if cfg.get('use_pair') else np.zeros(int(cfg.get('pair_dim',4)));temporal=float(np.dot(history_sender,history_candidate)/dim);graph=float(np.dot(sender,candidate_embedding)/dim) if cfg.get('use_gnn') else 0.0
        recipient[candidate]=float(cfg.get('recipient_pair_weight',0))*math.log1p(state['pairs'][u,candidate])+float(cfg.get('recipient_activity_weight',0))*math.log1p(state['ic'][candidate])+float(cfg.get('recipient_temporal_weight',0))*temporal+float(cfg.get('recipient_pair_state_weight',0))*(pair[0] if len(pair) else 0)+float(cfg.get('recipient_graph_weight',0))*graph
    amount=sequence_amount_logits(model,state,u,v,t);gap_logits=sequence_gap_logits(model,state,u,v,t);probabilities=[sequence_softmax(recipient),sequence_softmax(amount),sequence_softmax(gap_logits)];gap=max(0.0,t-state['last'][u]) if state['seen'][u] else None;buckets=[v,int(np.searchsorted(model['amount_bins'],e['amount'],side='right')),int(np.searchsorted(model['gap_bins'],gap,side='right')) if gap is not None else len(model['gap_bins'])+1];parts=[-math.log2(max(1e-30,float(probabilities[i][buckets[i]]))) for i in range(3)]
    return {'score':float(sum(parts)),'parts':parts,'probabilities':probabilities,'buckets':buckets,'gap':gap,'embedding':[sender,sequence_account_embedding(model,state,v,t,cache)],'features':xgb_feature_row(state,e).tolist()}

def sequence_supervised_rows(model,dataset):
    rows=[];labels=[];episodes=[]
    for episode_index,episode in enumerate(dataset['episodes']):
        state=xgb_state(episode['accounts'])
        for e in episode['events']:
            if e['kind']=='payment' and int(e.get('label',-1))>=0:
                prediction=sequence_prediction(model,state,e);rows.append(np.asarray(prediction['features']+[prediction['score']],dtype=float));labels.append(int(e['label']));episodes.append(episode_index)
            xgb_apply(state,e)
    return np.asarray(rows,dtype=float),np.asarray(labels,dtype=float),np.asarray(episodes,dtype=int)

def fit_sequence_supervised_head(model,dataset,steps=220,learning_rate=.08):
    X,y,episode_ids=sequence_supervised_rows(model,dataset);split=max(1,int(len(dataset['episodes'])*.8));train=episode_ids<split;valid=~train
    if train.sum()==0 or y[train].sum()==0:raise ValueError('Supervised mode requires at least one flagged fraud example.')
    mean=X[train].mean(0);scale=X[train].std(0);scale=np.where(scale<1e-8,1.0,scale);Xs=(X[train]-mean)/scale;yt=y[train];positive=max(1.0,float(yt.sum()));negative=max(1.0,float(len(yt)-yt.sum()));pos_weight=min(10.0,negative/positive);weights=np.where(yt>0,pos_weight,1.0);rate=float(np.clip(yt.mean(),1e-4,1-1e-4));w=np.zeros(Xs.shape[1]);b=math.log(rate/(1-rate))
    for _ in range(int(steps)):
        p=1/(1+np.exp(-np.clip(Xs@w+b,-50,50)));error=(p-yt)*weights;w-=learning_rate*(Xs.T@error/max(1,len(yt)));b-=learning_rate*float(error.mean())
    raw=(X[valid]-mean)/scale@w+b;vp=1/(1+np.exp(-np.clip(raw,-50,50)));vy=y[valid];val_loss=float(-(vy*np.log(np.maximum(vp,1e-12))+(1-vy)*np.log(np.maximum(1-vp,1e-12))).mean())
    return {'type':'logistic_fraud_head','mean':mean.tolist(),'scale':scale.tolist(),'weights':w.tolist(),'bias':float(b),'training':{'fraud_labels_used':int(yt.sum()),'fraud_flags_requested':int(dataset.get('flags_requested',0)),'validation_flags_used':int(dataset.get('validation_flags_used',0)),'training_rows':int(train.sum()),'validation_rows':int(valid.sum()),'steps':int(steps),'learning_rate':learning_rate,'scale_pos_weight':pos_weight,'heldout_logloss':val_loss,'notice':'Supervised head trained offline from flagged historical outcomes; scores are not calibrated production fraud probabilities.'}}

def sequence_trace(model,episode,head=None):
    state=xgb_state(episode['accounts']);records=[];empty=[[[] for _ in range(episode['accounts'])]]
    for e in episode['events']:
        before=[[] for _ in range(episode['accounts'])]
        if e['kind']=='payment':
            prediction=sequence_prediction(model,state,e);base=prediction['score']
            if head is not None:
                row=prediction['features']+[base];p=supervised_probability(head,row);score=-math.log2(max(1e-12,1-p));probabilities=[[[1-p,p]],[[1.0]],[[1.0]]]
            else:score=base;probabilities=[[prediction['probabilities'][i].tolist()] for i in range(3)]
            records.append({'scores':[score],'probabilities':probabilities,'memory_before':[before],'embedding':[prediction['embedding'][0].tolist(),prediction['embedding'][1].tolist()]})
        else:records.append({'scores':[0.0],'probabilities':[[[1.0]],[[1.0]],[[1.0]]],'memory_before':[before],'embedding':[[],[]]})
        xgb_apply(state,e);records[-1]['memory_after']=empty
    return records

def sequence_payload(a,design):
    variant=DESIGNS[design]['variant'];cfg=dict(SEQ_VARIANTS[variant]);cfg.update({'dim':8,'pair_dim':4,'history_limit':8,'pair_limit':6,'neighbor_limit':4,'gnn_layers':2,'history_decay':720.0,'pair_decay':1440.0,'pair_amount_weight':1.5,'pair_gap_weight':1.5,'histogram_smoothing':.5})
    seed=a.model_seed+sum((i+1)*ord(c) for i,c in enumerate(design));rng=np.random.default_rng(seed);cfg['history_weights']=rng.normal(0,.24,(8,8)).tolist();cfg['pair_weights']=rng.normal(0,.24,(4,5)).tolist();cfg['gnn_weights']=rng.normal(0,.18,(8,12)).tolist()
    return {'version':3,'id':design,'adapter':'temporal_family','family':'temporal_family','label':DESIGNS[design]['label'],'architecture':DESIGNS[design],'hidden':0,'neighbors':int(cfg['neighbor_limit']),'amount_bins':AMOUNT,'gap_bins':GAP,'sequence':cfg,'training':{'method':'Causal temporal-history likelihood with bounded sinusoidal time encoding','variant':variant,'fraud_labels_used':0,'default_warmup_requests':int(a.warmup),'model_seed':int(seed),'history_limit':int(cfg['history_limit']),'pair_limit':int(cfg['pair_limit']),'gnn_layers':int(cfg['gnn_layers'] if cfg.get('use_gnn') else 0),'notice':'No fraud labels enter the temporal-history score; optional flagged-history mode adds a separately trained offline fraud head.'},'policy':{'default_warmup_requests':int(a.warmup)}}

def train_sequence_design(a,data,design,labeled=None):
    payload=sequence_payload(a,design);supervised=fit_sequence_supervised_head(payload,labeled) if labeled is not None else None;payload['supervised']=supervised;cfg=payload['sequence'];payload['parameters']=sum(np.asarray(cfg[k]).size for k in ['history_weights','pair_weights','gnn_weights']);payload['checkpoint_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
    (ROOT/'models'/f'{design}.json').write_text(json.dumps(payload,separators=(',',':')));episode=data[-1];(ROOT/'parity'/f'{design}.json').write_text(json.dumps({'episode':episode,'expected':sequence_trace(payload,episode)},separators=(',',':')))
    if supervised is not None:(ROOT/'parity-supervised').mkdir(exist_ok=True);(ROOT/'parity-supervised'/f'{design}.json').write_text(json.dumps({'episode':episode,'expected':sequence_trace(payload,episode,supervised)},separators=(',',':')))
    print(design,'temporal-history checkpoint','parameters',payload['parameters'],'flags',supervised['training']['fraud_labels_used'] if supervised else 0,flush=True)

def state_memory(state):
    return [[] for _ in range(len(state['last']))]

def ensure_labeled_data(flags,seed):
    path=ROOT/'xgb-training.json';expected={'version':3,'flags_requested':int(flags),'seed':int(seed)}
    if path.exists():
        try:
            current=json.loads(path.read_text())
            if all(current.get(k)==v for k,v in expected.items()):return current
        except (json.JSONDecodeError,OSError):pass
    script="const fs=require('fs'),s=require('./scenarios');fs.writeFileSync('xgb-training.json',JSON.stringify(s.trainingLabeled(%d,%d)));"%(int(flags),int(seed))
    subprocess.run(['node','-e',script],cwd=ROOT,check=True)
    return json.loads(path.read_text())

def train_xgboost(a):
    data=ensure_labeled_data(a.fraud_flags,a.fraud_seed);X,y,episode_ids=xgb_rows(data);split=max(1,int(len(data['episodes'])*.8));train=episode_ids<split;valid=~train
    booster=fit_xgb(X[train],y[train],trees=a.xgb_trees,depth=3,learning_rate=.08,min_child=12,reg_lambda=1.0,gamma=.0,seed=a.model_seed)
    vp=1/(1+np.exp(-np.clip(xgb_predict_raw(booster,X[valid]),-50,50)));vy=y[valid]
    val_nll=float(-(vy*np.log(np.maximum(vp,1e-12))+(1-vy)*np.log(np.maximum(1-vp,1e-12))).mean())
    weights=json.dumps(booster,sort_keys=True,separators=(',',':'))
    summary={'method':'XGBoost-compatible second-order binary logistic tree booster','implementation':'Pure NumPy export; browser evaluates the exported trees without Python or xgboost runtime.','fraud_labels_used':int(y[train].sum()),'fraud_flags_requested':int(a.fraud_flags),'validation_flags_used':int(data.get('validation_flags_used',0)),'unknown_fraud_events_excluded':int(data.get('unknown_fraud_events',0)),'fraud_seed':int(a.fraud_seed),'default_warmup_requests':int(a.warmup),'model_seed':int(a.model_seed),'trees':int(a.xgb_trees),'max_depth':3,'learning_rate':.08,'scale_pos_weight':booster['scale_pos_weight'],'training_rows':int(train.sum()),'validation_rows':int(valid.sum()),'heldout_logloss':val_nll,'notice':'Synthetic flagged outcomes are an offline training signal; this is not calibrated production fraud probability.'}
    payload={'version':3,'id':'xgboost','adapter':'xgboost','family':'xgboost','label':DESIGNS['xgboost']['label'],'architecture':DESIGNS['xgboost'],'hidden':0,'neighbors':0,'amount_bins':AMOUNT,'gap_bins':GAP,'features':XGB_FEATURES,'base_score':booster['base_score'],'learning_rate':booster['learning_rate'],'trees':booster['trees'],'training':summary,'policy':{'default_warmup_requests':int(a.warmup)},'parameters':int(len(booster['trees'])),'checkpoint_id':hashlib.sha256(weights.encode()).hexdigest()[:16]}
    (ROOT/'models/xgboost.json').write_text(json.dumps(payload,separators=(',',':')))
    episode=data['episodes'][-1]
    # Keep separate parity fixtures for the two selectable training modes:
    # no-label mode uses the causal rarity score, while supervised mode uses
    # the exported fraud booster probability.
    (ROOT/'parity/xgboost.json').write_text(json.dumps({'episode':episode,'expected':xgb_unsupervised_trace(episode)},separators=(',',':')))
    (ROOT/'parity-supervised').mkdir(exist_ok=True)
    (ROOT/'parity-supervised/xgboost.json').write_text(json.dumps({'episode':episode,'expected':xgb_trace(booster,episode)},separators=(',',':')))
    print('xgboost held-out flagged logloss',val_nll,'parameters',payload['parameters'],'flags',int(y[train].sum()),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--design',choices=['all']+list(DESIGNS),default='all');ap.add_argument('--steps','--training-steps',dest='steps',type=int,default=140);ap.add_argument('--batch',type=int,default=12);ap.add_argument('--model-seed',type=int,default=512);ap.add_argument('--fraud-flags',type=int,default=int(CATALOG.get('default_fraud_flags',96)),help='Flagged historical fraud outcomes used by the offline XGBoost trainer.');ap.add_argument('--fraud-seed',type=int,default=811);ap.add_argument('--xgb-trees',type=int,default=32);ap.add_argument('--warmup','--calibration-requests','--training-time',dest='warmup',type=int,default=int(CATALOG.get('default_warmup',128)),help='Default number of unlabeled request scores used to initialize the live cutoff.');a=ap.parse_args()
    if a.steps<1 or a.batch<1 or a.fraud_flags<0 or a.xgb_trees<1 or a.warmup<1:raise SystemExit('steps, batch, fraud-flags, xgb-trees, and warmup must be positive (fraud-flags may be zero only when not training XGBoost).')
    if not (ROOT/'training-data.json').exists():subprocess.run(['node','-e',"require('fs').writeFileSync('training-data.json',JSON.stringify(require('./scenarios.js').trainingEpisodes()))"],cwd=ROOT,check=True)
    for folder in ['models','parity','parity-supervised']:(ROOT/folder).mkdir(exist_ok=True)
    data=json.loads((ROOT/'training-data.json').read_text())
    designs=list(DESIGNS)if a.design=='all'else [a.design]
    labeled=ensure_labeled_data(a.fraud_flags,a.fraud_seed) if a.fraud_flags>0 and any(DESIGNS[d].get('family')!='xgboost' for d in designs) else None
    for design in designs:
        if DESIGNS[design].get('family')=='xgboost':
            if a.fraud_flags<1:raise SystemExit('XGBoost requires at least one --fraud-flags example.')
            train_xgboost(a)
        elif DESIGNS[design].get('family')=='temporal_family':train_sequence_design(a,data,design,labeled)
        else:train_design(a,data,design,labeled)
if __name__=='__main__':main()
