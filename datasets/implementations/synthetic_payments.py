"""Access the existing deterministic generator as an isolated dataset provider."""
import json,os,subprocess
from framework.contracts import EventDataset
from framework.registry import ROOT

def load(config,base):
    options=config.get('options',{})
    script="const s=require('./datasets/implementations/synthetic_payments');const fs=require('fs');const o=JSON.parse(fs.readFileSync(0,'utf8'));if(!s.catalog.some(x=>x.id===(o.name||'relay')))throw Error('Unknown scenario');process.stdout.write(JSON.stringify(s.build(o.name||'relay',o.size||'medium',o.seed??42)));"
    result=subprocess.run([os.environ.get('NODE_BINARY','node'),'-e',script],input=json.dumps(options),cwd=ROOT,text=True,capture_output=True,check=True)
    document=json.loads(result.stdout);document['schema']='payment-events/v1';document['units']={'time':'minutes','currency':'EUR'};metadata={'loader':'synthetic_payments','origin':'synthetic','configuration':config};document['provenance']=metadata
    return EventDataset(document,metadata)
