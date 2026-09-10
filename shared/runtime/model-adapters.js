/* Compatibility bootstrap. Individual models are discovered from their manifest. */
(function(global){
  'use strict';
  if(typeof module!=='undefined'){
    const registry=require('./model-registry');
    for(const entry of require('../../models/registry.json').models)if(entry.browser)require('../../'+entry.browser);
    module.exports=registry;
  }else if(!global.FraudAdapters)throw Error('Model registry is not loaded.');
})(globalThis);
