/* GRU memory. See matching Python module for training. */
(function(global){
'use strict';
const registry=typeof module!=='undefined'?require('../../shared/runtime/model-registry'):global.FraudAdapters;
const base=typeof module!=='undefined'?require('./_neural'):global.FraudNeuralPrimitives;
// Architecture: gru memory; none readout; 0 graph layers.
const adapter={initialize:base.initialize,update:base.update,readout:base.readout,predict:base.predict};
registry.register('gru',adapter);
if(typeof module!=='undefined')module.exports=adapter;
})(globalThis);
