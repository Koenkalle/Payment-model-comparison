"""Shared payment feature constants and historical training-data access."""
import json,math,hashlib,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
H,K=8,5
AMOUNT=[15,35,75,150,300,650,1500,4000,10000]
GAP=[1,5,20,60,180,720,2880]
from datasets.payment_features import FEATURE_NAMES as XGB_FEATURES
CATALOG=json.loads((ROOT/'designs.json').read_text())
DESIGNS={d['id']:d for d in CATALOG['designs']}
SEQ_VARIANTS={
    'dygformer':{'use_pair':False,'use_gnn':False,'recipient_pair_weight':.4,'recipient_activity_weight':.2,'recipient_temporal_weight':1.2,'recipient_pair_state_weight':0.0,'recipient_graph_weight':0.0},
    'tami':{'use_pair':True,'use_gnn':False,'recipient_pair_weight':1.2,'recipient_activity_weight':.2,'recipient_temporal_weight':.25,'recipient_pair_state_weight':1.0,'recipient_graph_weight':0.0},
    'dyg_tami':{'use_pair':True,'use_gnn':False,'recipient_pair_weight':.8,'recipient_activity_weight':.2,'recipient_temporal_weight':.8,'recipient_pair_state_weight':1.0,'recipient_graph_weight':0.0},
    'dyg_tami_gnn':{'use_pair':True,'use_gnn':True,'recipient_pair_weight':.65,'recipient_activity_weight':.15,'recipient_temporal_weight':.6,'recipient_pair_state_weight':.8,'recipient_graph_weight':.8}
}


def ensure_labeled_data(flags,seed):
    path=ROOT/'xgb-training.json';expected={'version':3,'flags_requested':int(flags),'seed':int(seed)}
    if path.exists():
        try:
            current=json.loads(path.read_text())
            if all(current.get(k)==v for k,v in expected.items()):return current
        except (json.JSONDecodeError,OSError):pass
    script="const fs=require('fs'),s=require('./shared/runtime/scenarios');fs.writeFileSync('xgb-training.json',JSON.stringify(s.trainingLabeled(%d,%d)));"%(int(flags),int(seed))
    subprocess.run(['node','-e',script],cwd=ROOT,check=True)
    return json.loads(path.read_text())
