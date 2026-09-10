/* Local payment-event loader. Outcomes are removed from scoring events. */
(function(global){
  'use strict';const registry=typeof module!=='undefined'?require('../../shared/runtime/dataset-registry'):global.FraudDatasets;
  function load(input){
    const raw=typeof input==='string'?JSON.parse(input):input;
    if(!raw||typeof raw!=='object'||(raw.schema&&raw.schema!=='payment-events/v1'))throw Error('Expected a payment-events/v1 document.');
    const units=raw.units||{time:'minutes',currency:'EUR'};
    if(units.time!=='minutes'||units.currency!=='EUR')throw Error('Convert payment data to minutes and EUR before importing.');
    if(!Array.isArray(raw.accounts)||!raw.accounts.length||!Array.isArray(raw.events)||!raw.events.length)throw Error('A dataset needs nonempty accounts and events.');
    if(raw.accounts.length>256||raw.events.length>20000)throw Error('The browser replay supports at most 256 accounts and 20,000 events. Use a bounded event subset or the Python experiment runner.');
    const external=new Set(),accounts=raw.accounts.map((account,index)=>{if(account.id!==index)throw Error('Account IDs must be contiguous indices; normalize external IDs with the CSV loader.');const id=String(account.external_id??index);if(external.has(id))throw Error('Duplicate external account ID.');external.add(id);return {...account,external_id:id,name:String(account.name??id)};});
    const ids=new Set(),payments=new Set();let last=-Infinity;
    const events=raw.events.map(rawEvent=>{
      const {id,kind,t,u,v,amount}=rawEvent;
      if(typeof id!=='string'||!id||ids.has(id))throw Error('Event IDs must be nonempty and unique.');ids.add(id);
      if(!Number.isFinite(t)||t<0||t<last||!Number.isFinite(amount)||amount<0)throw Error('Events must have finite nonnegative amounts and chronological times.');last=t;
      if(!Number.isInteger(u)||!Number.isInteger(v)||v<0||v>=accounts.length||u< -1||u>=accounts.length||u===v)throw Error('Invalid account index.');
      if(!['payment','deposit','report'].includes(kind)||(kind==='payment')!==(u>=0))throw Error('Invalid event kind/sender combination.');
      if(kind==='payment')payments.add(id);
      const event={id,kind,t,u,v,amount};
      if(rawEvent.settled!==undefined){if(typeof rawEvent.settled!=='boolean')throw Error('settled must be boolean.');event.settled=rawEvent.settled;}
      if(rawEvent.reference!==undefined)event.reference=String(rawEvent.reference);
      return event;
    });
    const truth={};for(const [id,value]of Object.entries(raw.truth||{})){if(!payments.has(id))throw Error('Outcome refers to an unknown payment: '+id);if(value===true||value===1)truth[id]=true;else if(value===false||value===0)truth[id]=false;else if(![null,-1,'unknown',''].includes(value))throw Error('Outcomes must be 0, 1 or unknown.');}
    return {schema:'payment-events/v1',units:{...units},name:String(raw.name||'Imported payments'),size:'imported',seed:null,accounts,events,truth,focus:[],bookmarks:[],startIndex:0,description:String(raw.description||'User-supplied payment events.'),provenance:JSON.parse(JSON.stringify(raw.provenance||{origin:'user-supplied'}))};
  }
  registry.register('payment_json',{load});if(typeof module!=='undefined')module.exports={load};
})(globalThis);
