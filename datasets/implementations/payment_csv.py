"""Map real payment CSV columns to the shared chronological event schema."""
from framework.contracts import EventDataset
from ._validation import source,provenance,rows,number,label,timestamp
from .payment_json import normalize

def load(config,base):
    path=source(config,base);columns=config['columns']
    required=['id','time','sender','recipient','amount']
    if any(key not in columns for key in required):raise ValueError('columns must map id, time, sender, recipient and amount.')
    if config.get('currency','EUR')!='EUR':raise ValueError('Explicit currency conversion is required before loading non-EUR payments.')
    data=rows(path,list(columns.values()));accounts=[];mapping={};events=[];truth={};availability={}
    def account(value):
        value=str(value).strip()
        if not value:raise ValueError('Missing account identity.')
        if value not in mapping:
            mapping[value]=len(accounts);accounts.append({'id':mapping[value],'external_id':value,'name':value})
        return mapping[value]
    times=[timestamp(row[columns['time']],config.get('time_unit','seconds')) for row in data];origin=min(times)
    for row,time in zip(data,times):
        kind=row[columns['kind']].strip() if 'kind' in columns else 'payment'
        event={'id':row[columns['id']].strip(),'kind':kind,'t':(time-origin)/60,'u':account(row[columns['sender']]) if kind=='payment' else -1,'v':account(row[columns['recipient']]),'amount':number(row[columns['amount']],'amount')}
        if 'settled' in columns:
            value=row[columns['settled']].strip().lower()
            if value not in ('true','false','0','1'):raise ValueError('settled CSV values must be true/false or 1/0.')
            event['settled']=value in ('true','1')
        if 'reference' in columns:event['reference']=row[columns['reference']]
        if 'label' in columns and kind=='payment':
            outcome=label(row[columns['label']],config.get('label_values'))
            if outcome>=0:
                truth[event['id']]=bool(outcome)
                if 'label_available_at' in columns and row[columns['label_available_at']].strip():
                    availability[event['id']]=(timestamp(row[columns['label_available_at']],config.get('time_unit','seconds'))-origin)/60
        events.append(event)
    events.sort(key=lambda event:event['t'])
    metadata=provenance(config,path);metadata['time_origin_unix_or_numeric_seconds']=origin
    raw={'schema':'payment-events/v1','name':config.get('name',path.stem),'accounts':accounts,'events':events,'truth':truth,'units':{'time':'minutes','currency':'EUR'}}
    if availability:raw['label_available_at']=availability
    document=normalize(raw,metadata)
    return EventDataset(document,metadata)
