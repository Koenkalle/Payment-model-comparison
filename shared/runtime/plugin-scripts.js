/* Shared manifest expansion for Node tools and browser-test harnesses. */
'use strict';
const models=require('../../models/registry.json'),datasets=require('../../datasets/registry.json');
module.exports=()=>[...new Set([...models.browser_support,...models.models.filter(m=>m.browser).map(m=>m.browser),...datasets.browser_support,...datasets.datasets.filter(d=>d.browser).map(d=>d.browser)])];
