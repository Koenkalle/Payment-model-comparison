/* Temporal + pair prototype. See matching Python module for training. */
(function(global){
'use strict';
const registry=typeof module!=='undefined'?require('../../shared/runtime/model-registry'):global.FraudAdapters;
const base=typeof module!=='undefined'?require('./_temporal'):global.FraudTemporalPrototype;
// The checkpoint config selects pair history and graph aggregation; projections stay fixed.
const adapter={...base};
registry.register('dyg_tami',adapter);
if(typeof module!=='undefined')module.exports=adapter;
})(globalThis);
