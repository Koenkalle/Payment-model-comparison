/* Dataset loading contract. The runner receives normalized data, never filenames. */
(function(global){
  'use strict';const entries=new Map();
  function register(id,loader){if(entries.has(id)||typeof loader.load!=='function')throw Error('Invalid dataset implementation: '+id);entries.set(id,loader);}
  function load(id,input){if(!entries.has(id))throw Error('Dataset loader is not available: '+id);return entries.get(id).load(input);}
  global.FraudDatasets={register,load,list:()=>Array.from(entries.keys())};
  if(typeof module!=='undefined')module.exports=global.FraudDatasets;
})(globalThis);
