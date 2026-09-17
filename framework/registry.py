"""Manifest-based discovery, with lazy imports and explicit execution capabilities."""
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def manifest(kind):
    if kind not in ('models','datasets'):raise ValueError('Unknown plugin kind: '+kind)
    payload=json.loads((ROOT/kind/'registry.json').read_text())
    if payload.get('version')!=1:raise ValueError('Unsupported registry version.')
    entries=payload[kind]
    ids=[entry['id'] for entry in entries]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate plugin IDs.')
    return payload

def entry(kind,identifier):
    found=next((item for item in manifest(kind)[kind] if item['id']==identifier),None)
    if found is None:raise ValueError(f'Unknown {kind} implementation: {identifier}')
    return found

def model_entry(identifier):return entry('models',identifier)
def dataset_entry(identifier):return entry('datasets',identifier)

def load_module(name):
    if not name.startswith(('models.implementations.','datasets.implementations.')):
        raise ValueError('Plugin modules must be declared under their implementation package.')
    return importlib.import_module(name)

def create_model(identifier,input_schema,task=None):
    descriptor=model_entry(identifier)
    if task is not None:
        selected=descriptor.get('tasks',{}).get(task)
        if selected is not None:
            descriptor={**descriptor,**selected,'capabilities':{**descriptor.get('capabilities',{}),'task':task}}
        elif descriptor.get('capabilities',{}).get('task')!=task:
            raise ValueError(f'{identifier} does not support task {task}.')
    if 'python' not in descriptor['execution'] or input_schema not in descriptor['inputs']:
        raise ValueError(f'{identifier} does not support Python experiments with {input_schema}; capabilities: {descriptor["execution"]}, {descriptor["inputs"]}')
    return load_module(descriptor['python_module']).create(),descriptor

def load_dataset(config,base_dir=None):
    descriptor=dataset_entry(config['loader'])
    expected=descriptor['schema']
    if descriptor.get('views'):
        view=config.get('view',descriptor.get('default_view','stream'))
        if view not in descriptor['views']:raise ValueError('Unsupported dataset view: '+str(view))
        expected=descriptor['views'][view]
    dataset=load_module(descriptor['python_module']).load(config,Path(base_dir or '.').resolve())
    if dataset.schema!=expected:raise ValueError('Dataset implementation returned the wrong schema.')
    return dataset

def browser_scripts():
    model_manifest=manifest('models');dataset_manifest=manifest('datasets')
    scripts=model_manifest['browser_support']+[e['browser'] for e in model_manifest['models'] if e.get('browser')]
    scripts+=dataset_manifest['browser_support']+[e['browser'] for e in dataset_manifest['datasets'] if e.get('browser')]
    return list(dict.fromkeys(scripts))
