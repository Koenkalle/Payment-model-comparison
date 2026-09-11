/* Compatibility bootstrap. Individual models are discovered from their manifest. */
(function(global){
  'use strict';
  if(typeof module!=='undefined'){
    const registry=require('./model-registry');
    const manifest=require('../../models/registry.json');
    for(const source of manifest.browser_support)require('../../'+source);
    for(const entry of manifest.models)if(entry.browser)require('../../'+entry.browser);
    module.exports=registry;
  }else if(!global.FraudAdapters)throw Error('Model registry is not loaded.');
})(globalThis);
