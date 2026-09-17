/* Dataset-owned payment features; the describe adapter preserves settled-history
 * checkpoint inputs. New materialized datasets use observed attempts and strict
 * timestamp groups in datasets/features.py. */
(function(global){
  'use strict';
  const DEFAULT_AMOUNT_BINS=[15,35,75,150,300,650,1500,4000,10000];
  const featureLog=(x,scale=1)=>Math.log1p(Math.max(0,Number(x)||0))/scale;
  const specs=[
    ['log_amount','Payment amount','currency units',8,'Size of the current requested payment, compressed with the saved logarithmic transformation so large transfers do not dominate the numeric scale.','payment','current payment'],
    ['amount_bin','Amount bucket','bucket',null,'Where the requested payment amount falls among the checkpoint’s amount boundaries. Its zero-based bucket counts boundaries less than or equal to the amount; payments in the same bucket receive the same value.','payment','current payment'],
    ['sender_out_count','Sender outgoing payments','payments',6,'How many earlier settled payments this sender sent. Repeated payments to the same recipient count separately.','sender','all preceding history'],
    ['sender_in_count','Sender incoming payments','payments',6,'How many earlier settled payments this sender received. Repeated payments from the same counterparty count separately.','sender','all preceding history'],
    ['recipient_out_count','Recipient outgoing payments','payments',6,'How many earlier settled payments this recipient sent. This measures the recipient’s own outgoing activity, not payments received from the current sender.','recipient','all preceding history'],
    ['recipient_in_count','Recipient incoming payments','payments',6,'How many earlier settled payments this recipient received. Payments from all counterparties contribute, not just those from the current sender.','recipient','all preceding history'],
    ['sender_out_value','Sender outgoing amount','currency units',10,'Total amount this sender sent through earlier settled payments. This measures accumulated outgoing value rather than the number of payments.','sender','all preceding history'],
    ['sender_in_value','Sender incoming amount','currency units',10,'Total amount this sender received through earlier settled payments. This measures accumulated incoming value rather than the number of payments.','sender','all preceding history'],
    ['recipient_out_value','Recipient outgoing amount','currency units',10,'Total amount this recipient sent through earlier settled payments. This describes the recipient’s outgoing value to all counterparties.','recipient','all preceding history'],
    ['recipient_in_value','Recipient incoming amount','currency units',10,'Total amount this recipient received through earlier settled payments. This describes incoming value from all counterparties, not just the current sender.','recipient','all preceding history'],
    ['sender_seen','Sender observed activities','activities',6,'How much earlier activity has been observed for this sender, including payments sent, payments received and deposits. Unsettled attempts count; outcome reports do not.','sender','all preceding history'],
    ['recipient_seen','Recipient observed activities','activities',6,'How much earlier activity has been observed for this recipient, including payments sent, payments received and deposits. Unsettled attempts count; outcome reports do not.','recipient','all preceding history'],
    ['sender_gap','Sender activity gap','minutes',6,'How long this sender has been inactive: nonnegative minutes since its last observed payment or deposit. First activity gives zero, and outcome reports do not reset the gap.','sender','last observed activity'],
    ['recipient_gap','Recipient activity gap','minutes',6,'How long this recipient has been inactive: nonnegative minutes since its last observed payment or deposit. First activity gives zero, and outcome reports do not reset the gap.','recipient','last observed activity'],
    ['pair_out_count','Prior payments to recipient','payments',3,'Earlier settled payments from the current sender to this particular recipient. Payments in the reverse direction do not contribute.','account pair','all preceding history'],
    ['pair_total_count','Prior payments in either direction','payments',3,'Earlier settled payments between the current sender and recipient, counting both directions. Repeated payments contribute separately.','account pair','all preceding history'],
    ['prior_contact','Prior payment contact','boolean',null,'Whether the current sender and recipient have paid each other before: one for an earlier settled payment in either direction, zero when no such contact exists.','account pair','all preceding history'],
    ['sender_recent_in_count','Sender recent incoming payments','payments',4,'How many earlier settled payments this sender received in the preceding 60 minutes. This measures recent incoming activity and includes the window boundary.','sender','preceding 60 minutes'],
    ['sender_recent_out_count','Sender recent outgoing payments','payments',4,'How many earlier settled payments this sender sent in the preceding 60 minutes. This measures recent outgoing activity and includes the window boundary.','sender','preceding 60 minutes'],
    ['recipient_recent_in_count','Recipient recent incoming payments','payments',4,'How many earlier settled payments this recipient received in the preceding 60 minutes. Payments from all counterparties contribute, including at the window boundary.','recipient','preceding 60 minutes'],
    ['recipient_recent_out_count','Recipient recent outgoing payments','payments',4,'How many earlier settled payments this recipient sent in the preceding 60 minutes. This measures the recipient’s recent outgoing activity and includes the window boundary.','recipient','preceding 60 minutes'],
    ['sender_recent_in_value','Sender recent incoming amount','currency units',10,'Total amount this sender received through earlier settled payments in the preceding 60 minutes, including the window boundary. This measures recent incoming value.','sender','preceding 60 minutes'],
    ['sender_recent_out_value','Sender recent outgoing amount','currency units',10,'Total amount this sender sent through earlier settled payments in the preceding 60 minutes, including the window boundary. This measures recent outgoing value.','sender','preceding 60 minutes'],
    ['recipient_recent_in_value','Recipient recent incoming amount','currency units',10,'Total amount this recipient received through earlier settled payments in the preceding 60 minutes, including the window boundary. Incoming payments from all counterparties contribute.','recipient','preceding 60 minutes'],
    ['recipient_recent_out_value','Recipient recent outgoing amount','currency units',10,'Total amount this recipient sent through earlier settled payments in the preceding 60 minutes, including the window boundary. This measures the recipient’s recent outgoing value.','recipient','preceding 60 minutes'],
    ['amount_vs_sender_out_mean','Amount / sender outgoing mean','ratio',8,'How large this payment is relative to amounts this sender previously sent through settled payments. Count and mean denominators are floored at one; without settled history the raw ratio equals the amount.','payment and sender','all preceding history'],
    ['amount_vs_recipient_in_mean','Amount / recipient incoming mean','ratio',8,'How large this payment is relative to amounts this recipient previously received through settled payments. Count and mean denominators are floored at one; without settled history the raw ratio equals the amount.','payment and recipient','all preceding history']
  ];
  const featureDefinitions=Object.freeze(specs.map(([id,label,unit,scale,description,source,window],index)=>Object.freeze({id,label,unit,description,source,window,index,transform:scale?'log1p(max(0, value)) / '+scale:id==='amount_bin'?'bucket index / number of amount-bin edges':'identity',scale})));
  function recent(s,n,role,t){const events=s.incidents[n].filter(x=>t-x.t<=60&&x.role===role);return [events.length,events.reduce((sum,x)=>sum+x.amount,0)];}
  function describe(model,s,e){
    const u=e.u,v=e.v,t=e.t,amount=e.amount,si=recent(s,u,1,t),so=recent(s,u,-1,t),ri=recent(s,v,1,t),ro=recent(s,v,-1,t);
    const senderMean=s.outValue[u]/Math.max(1,s.outCount[u]),recipientMean=s.inValue[v]/Math.max(1,s.inCount[v]);
    const gapU=s.seen[u]?Math.max(0,t-s.last[u]):0,gapV=s.seen[v]?Math.max(0,t-s.last[v]):0;
    const amountBins=model.amount_bins||DEFAULT_AMOUNT_BINS;
    const readableValues=[amount,amountBins.reduce((n,edge)=>n+(amount>=edge),0),
      s.outCount[u],s.inCount[u],s.outCount[v],s.inCount[v],s.outValue[u],s.inValue[u],s.outValue[v],s.inValue[v],
      s.seen[u],s.seen[v],gapU,gapV,s.pairs[u][v],s.pairs[u][v]+s.pairs[v][u],s.pairs[u][v]+s.pairs[v][u]>0?1:0,
      si[0],so[0],ri[0],ro[0],si[1],so[1],ri[1],ro[1],amount/Math.max(1,senderMean),amount/Math.max(1,recipientMean)];
    const values=readableValues.map((value,i)=>i===1?value/amountBins.length:specs[i][3]?featureLog(value,specs[i][3]):value);
    return {values,readableValues};
  }
  const features=(model,s,e)=>describe(model,s,e).values;
  global.FraudPaymentFeatures={DEFAULT_AMOUNT_BINS,featureDefinitions,features,describe};
  if(typeof module!=='undefined')module.exports=global.FraudPaymentFeatures;
})(typeof globalThis!=='undefined'?globalThis:window);
