"""Load explicitly selected numeric features from a local CSV file."""
import numpy as np
from framework.contracts import NumericDataset
from ._validation import source,provenance,rows,number,label,timestamp

def load(config,base):
    path=source(config,base);features=config.get('features',[])
    if not features or len(set(features))!=len(features) or not all(isinstance(f,str) for f in features):raise ValueError('features must list distinct column names in model input order.')
    id_column,time_column=config['id_column'],config['time_column'];label_column=config.get('label_column')
    roles=[id_column,time_column]+([label_column] if label_column else [])
    if len(set(roles))!=len(roles):raise ValueError('ID, timestamp and outcome columns must be distinct.')
    if set(features)&{id_column,time_column,label_column}:raise ValueError('Identifiers, timestamps and outcomes cannot also be feature columns.')
    required=list(features)+[id_column,time_column]+([label_column] if label_column else [])
    data=rows(path,required);ids=[row[id_column].strip() for row in data]
    if any(not value for value in ids) or len(set(ids))!=len(ids):raise ValueError('Transaction IDs must be nonempty and unique.')
    times=np.asarray([timestamp(row[time_column],config.get('time_unit','seconds')) for row in data])
    values=np.asarray([[number(row[f],f) for f in features] for row in data],dtype=float)
    labels=np.asarray([label(row.get(label_column),config.get('label_values')) for row in data],dtype=int)
    order=np.argsort(times,kind='stable');times=times[order];values=values[order];labels=labels[order]
    for array in (times,values,labels):array.setflags(write=False)
    return NumericDataset(tuple(ids[i] for i in order),times,values,labels,tuple(features),provenance(config,path))
