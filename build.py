"""Assemble the self-contained, offline inference demo."""
from pathlib import Path
import json, subprocess

root=Path(__file__).resolve().parent
template=(root/'template.html').read_text()
catalog=json.loads((root/'designs.json').read_text())
models=[json.loads((root/'models'/f"{d['id']}.json").read_text())for d in catalog['designs']]
subprocess.run(['node','tune_policy.js','--if-stale'],cwd=root,check=True)
validation=json.loads((root/'policy-validation.json').read_text())
for model in models:
    model['policy_validation']=validation['models'].get(model['id'],{})
xgb=next((m for m in models if m.get('family')=='xgboost'),None)
bundle={'version':5,'default':catalog['default'],'policy':{'training_mode':'unsupervised','decision_policy':'shared','validation':validation['provenance'],'warmup':catalog.get('default_warmup',128),'fraud_flags':(xgb or {}).get('training',{}).get('fraud_flags_requested',catalog.get('default_fraud_flags',96))},'models':models}
(root/'model-bundle.json').write_text(json.dumps(bundle,separators=(',',':')))
data='<script type="application/json" id="fd-model-data">'+json.dumps(bundle,separators=(',',':')).replace('<','\\u003c')+'</script>\n'
split=template.rfind('</div>')
html=template[:split]+data+template[split:]
for name in ['d3.min.js','model-adapters.js','policy.js','model.js','scenarios.js','comparison.js']+catalog.get('runtime_scripts',[])+['ui.js']:
    html+='\n<script>\n'+(root/name).read_text()+'\n</script>\n'
destination=root.parent/'payment-model-comparison.html'
destination.write_text(html)
shell=root/'standalone-shell.html'
if shell.exists():
    (root/'index.html').write_text(shell.read_text().replace('<!-- FRAUD_DEMO_FRAGMENT -->',html))
print(destination)
