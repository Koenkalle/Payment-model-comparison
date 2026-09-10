"""Dataset-independent fitting, chronological validation, artifacts and evaluation."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from .contracts import NumericDataset,TemporalGraphDataset
from .registry import ROOT,create_model,load_dataset


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def chronological_split(dataset,config):
    train_fraction=config.get('train',.6);validation_fraction=config.get('validation',.2)
    if not 0<train_fraction<1 or not 0<validation_fraction<1 or train_fraction+validation_fraction>=1:raise ValueError('Train and validation fractions must be positive and leave a test partition.')
    unique=np.unique(dataset.times);a=int(len(unique)*train_fraction);b=int(len(unique)*(train_fraction+validation_fraction))
    if a<1 or b<=a or b>=len(unique):raise ValueError('Need enough distinct timestamps for nonempty train, validation and test partitions.')
    # Keep equal timestamps together; no random split or tie leakage.
    return {'train':np.flatnonzero(dataset.times<unique[a]),'validation':np.flatnonzero((dataset.times>=unique[a])&(dataset.times<unique[b])),'test':np.flatnonzero(dataset.times>=unique[b])}

def metrics(labels,probabilities,threshold):
    known=labels>=0;actual=labels[known];blocked=probabilities[known]>threshold
    tp=int(np.sum((actual==1)&blocked));fp=int(np.sum((actual==0)&blocked));fn=int(np.sum((actual==1)&~blocked));tn=int(np.sum((actual==0)&~blocked))
    return {'rows':len(labels),'known':int(known.sum()),'unknown':int((~known).sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else (0 if known.any() else None),'block_rate':float(np.mean(probabilities>threshold)) if len(labels) else None}

def choose_threshold(labels,probabilities):
    known=labels>=0;y=labels[known];p=probabilities[known]
    if set(y.tolist())!={0,1}:raise ValueError('Automatic threshold selection needs both known classes in validation; configure decision_threshold for a fixed cutoff.')
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order];tp=fp=0;fn=int(y.sum());best=(0.,0,float(p[0]));index=0
    while index<len(y):
        value=p[index]
        while index<len(y) and p[index]==value:
            if y[index]==1:tp+=1;fn-=1
            else:fp+=1
            index+=1
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        candidate=(f1,-index,float(p[index]) if index<len(y) else -float(np.finfo(float).eps))
        if candidate[:2]>best[:2]:best=candidate
    return best[2]

def checked_predictions(model,dataset,indices,explain=False):
    result=model.predict(dataset.features[indices],dataset.feature_names,explain)
    p=np.asarray(result.probabilities)
    if p.shape!=(len(indices),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Model must return one finite probability in [0,1] per input row.')
    return result

def report(model,dataset,indices,threshold,explain=False):
    prediction=checked_predictions(model,dataset,indices,explain)
    rows=[]
    for offset,index in enumerate(indices):
        row={'id':dataset.ids[index],'timestamp_seconds':float(dataset.times[index]),'probability':float(prediction.probabilities[offset]),'label':int(dataset.labels[index]) if dataset.labels[index]>=0 else None,'decision':'BLOCK' if prediction.probabilities[offset]>threshold else 'ALLOW'}
        if prediction.margins is not None:row['margin']=float(prediction.margins[offset])
        if prediction.contributions is not None:row['contributions']=prediction.contributions[offset,:-1].tolist();row['baseline']=float(prediction.contributions[offset,-1])
        rows.append(row)
    return {'version':1,'schema':'model-evaluation/v1','feature_names':list(dataset.feature_names),'threshold':threshold,'dataset':dataset.provenance,'metrics':metrics(dataset.labels[indices],prediction.probabilities,threshold),'rows':rows}

def train_experiment(config,output,base_dir=None):
    dataset=load_dataset(config['dataset'],base_dir)
    if isinstance(dataset,TemporalGraphDataset):
        from .temporal_experiments import train
        return train(config,output,dataset)
    model,descriptor=create_model(config['model'],dataset.schema)
    if not isinstance(dataset,NumericDataset):raise ValueError('The experiment runner requires a numeric feature dataset.')
    output=Path(output).resolve()
    if output.exists():raise ValueError('Experiment output already exists; choose a new directory.')
    splits=chronological_split(dataset,config.get('split',{}));training=splits['train'];known=training[dataset.labels[training]>=0]
    if set(dataset.labels[known].tolist())!={0,1}:raise ValueError('Training requires known examples from both classes.')
    parameters=dict(config.get('parameters',{}))
    model.fit(dataset.features[known],dataset.labels[known],dataset.feature_names,parameters)
    validation=checked_predictions(model,dataset,splits['validation'])
    threshold=config.get('decision_threshold')
    if threshold is None:threshold=choose_threshold(dataset.labels[splits['validation']],validation.probabilities)
    elif not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('decision_threshold must be a probability between 0 and 1.')
    explained=bool(descriptor['capabilities'].get('explanations'))
    results=report(model,dataset,splits['test'],threshold,explained)
    results['partition']='test';results['training_rows_in_evaluation']=0
    implementation=ROOT/(descriptor['python_module'].replace('.','/')+'.py')
    metadata={'version':1,'model_id':descriptor['id'],'input_schema':dataset.schema,'feature_names':list(dataset.feature_names),'implementation':descriptor,'implementation_sha256':digest(implementation),'library_version':getattr(model,'library_version',None),'parameters':parameters,'dataset':dataset.provenance,'split':{name:[dataset.ids[i] for i in indices] for name,indices in splits.items()},'threshold':float(threshold),'threshold_source':'validation_f1' if config.get('decision_threshold') is None else 'configured','validation_metrics':metrics(dataset.labels[splits['validation']],validation.probabilities,threshold)}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.payment-experiment-',dir=output.parent) as temporary:
        stage=Path(temporary);model.save(stage/'model.json');metadata['model_sha256']=digest(stage/'model.json')
        (stage/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        (stage/'test-report.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        os.rename(stage,output)
    return metadata,results

def evaluate_artifact(artifact,dataset_config,base_dir=None,partition='all'):
    artifact=Path(artifact);metadata=json.loads((artifact/'manifest.json').read_text())
    if metadata.get('version')!=1:raise ValueError('Unsupported model artifact version.')
    if metadata.get('task')=='dynamic-link-prediction':
        from .temporal_experiments import evaluate
        return evaluate(artifact,dataset_config,base_dir,partition)
    if digest(artifact/'model.json')!=metadata['model_sha256']:raise ValueError('Model artifact checksum does not match.')
    dataset=load_dataset(dataset_config,base_dir)
    model,descriptor=create_model(metadata['model_id'],dataset.schema)
    if list(dataset.feature_names)!=metadata['feature_names']:raise ValueError('Dataset feature names or order differ from the fitted model.')
    implementation=ROOT/(descriptor['python_module'].replace('.','/')+'.py')
    if digest(implementation)!=metadata['implementation_sha256']:raise ValueError('Model implementation changed since this artifact was created.')
    model.load(artifact/'model.json',dataset.feature_names)
    if partition=='all':indices=np.arange(len(dataset.ids))
    elif partition in metadata['split']:
        if any(dataset.provenance[key]!=metadata['dataset'][key] for key in ('source_sha256','configuration_sha256')):raise ValueError('Saved split membership applies only to the original dataset.')
        wanted=set(metadata['split'][partition]);indices=np.asarray([i for i,value in enumerate(dataset.ids) if value in wanted])
        if len(indices)!=len(wanted):raise ValueError('Saved partition IDs are missing from this dataset.')
    else:raise ValueError('Unknown evaluation partition.')
    result=report(model,dataset,indices,metadata['threshold'],bool(descriptor['capabilities'].get('explanations')))
    result['partition']=partition
    same_source=dataset.provenance['source_sha256']==metadata['dataset']['source_sha256']
    trained=set(metadata['split']['train'])
    result['training_rows_in_evaluation']=sum(dataset.ids[i] in trained for i in indices) if same_source else None
    return result
