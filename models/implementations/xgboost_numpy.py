"""Custom NumPy booster used by existing demo checkpoints; not the XGBoost library."""
from ._common import *

def state_memory(state):
    return [[] for _ in range(len(state["last"]))]

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

def train_xgboost(a):
    if a.fraud_flags<1:raise ValueError('XGBoost requires at least one flagged training example.')
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


def train(args,data=None,labeled=None):
    return train_xgboost(args)
