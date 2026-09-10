/* Compatibility import. Dataset generation is implemented independently. */
(function(global){
 const api=typeof module!=='undefined'?require('../../datasets/implementations/synthetic_payments'):global.FraudScenarios;
 if(typeof module!=='undefined')module.exports=api;
})(globalThis);
