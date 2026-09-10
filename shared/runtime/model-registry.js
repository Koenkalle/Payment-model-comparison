/* Registration and lifecycle contract only. No model mathematics or family selection. */
(function(global){
  'use strict';
  const entries=new Map();
  function register(id,implementation){
    if(!id||entries.has(id))throw Error('Duplicate or invalid model implementation: '+id);
    for(const method of ['initialize','update','readout','predict'])if(typeof implementation[method]!=='function')throw Error(id+' must implement '+method+'().');
    entries.set(id,Object.freeze(implementation));
  }
  function get(checkpoint){
    const id=checkpoint.implementation_id||checkpoint.id||checkpoint.adapter;
    if(!entries.has(id))throw Error('Model implementation is not available in this runtime: '+id);
    return entries.get(id);
  }
  global.FraudAdapters={register,get,list:()=>Array.from(entries.keys()),architecture:model=>model.architecture||{memory:'gru',readout:'attention',layers:2}};
  if(typeof module!=='undefined')module.exports=global.FraudAdapters;
})(globalThis);
