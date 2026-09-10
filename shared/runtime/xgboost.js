/* Compatibility import; implementation and feature definitions are in models/implementations/. */
(function(global){
 const api=typeof module!=='undefined'?require('../../models/implementations/xgboost_numpy_core'):global.FraudXGBoost;
 if(!api)throw Error('XGBoost model implementation is not loaded.');
 if(typeof module!=='undefined')module.exports=api;
})(globalThis);
