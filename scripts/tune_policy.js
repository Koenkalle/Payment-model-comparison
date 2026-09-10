/* Offline score export: only the held-out historical episodes enter the fit. */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),core=require('../shared/runtime/model'),policy=require('../shared/runtime/policy');
const root=path.resolve(__dirname,'..'),read=name=>fs.readFileSync(path.join(root,name),'utf8'),sha=text=>crypto.createHash('sha256').update(text).digest('hex');
function sourceFiles(){return [...require('../shared/runtime/plugin-scripts')(),'models/registry.json','datasets/registry.json','scripts/tune_policy.js','shared/runtime/policy.js','shared/runtime/model.js','shared/runtime/model-adapters.js','shared/runtime/xgboost.js','designs.json','xgb-training.json',...JSON.parse(read('designs.json')).designs.map(d=>'models/'+d.id+'.json')];}
function signatures(){return Object.fromEntries(sourceFiles().map(name=>[name,sha(read(name))]));}
function validationRows(model,episodes,trainingMode){
  const rows=[];
  for(const episode of episodes){
    const state=core.initial(model,episode.accounts,{trainingMode});
    for(const raw of episode.events){
      // Outcome fields never enter the model, its memory or graph.
      const event={id:raw.id,t:raw.t,u:raw.u,v:raw.v,amount:raw.amount,kind:raw.kind};
      if(raw.settled!==undefined)event.settled=raw.settled;
      const prediction=core.score(model,state,event);
      if(prediction)rows.push({score:prediction.score,label:[0,1].includes(raw.label)?raw.label:-1});
      core.apply(model,state,event);
    }
  }
  return rows;
}
function generate(){
  const dataset=JSON.parse(read('xgb-training.json')),split=Math.max(1,Math.floor(dataset.episodes.length*.8)),heldout=dataset.episodes.slice(split),models={};
  const artifact={version:1,sources:signatures(),provenance:{dataset:'xgb-training.json',seed:dataset.seed,trainingEpisodeIndices:Array.from({length:split},(_,i)=>i),validationEpisodeIndices:heldout.map((_,i)=>split+i),protocol:'Held-out synthetic episodes; chronological score-before-update replay, observed settlements, no policy intervention. Unknown outcomes excluded from cost. Validation rates describe this sample only.'},models};
  for(const design of JSON.parse(read('designs.json')).designs){
    const model=JSON.parse(read('models/'+design.id+'.json'));models[model.id]={};
    for(const mode of['unsupervised','supervised']){
      if(mode==='supervised'&&!model.supervised&&model.family!=='xgboost')continue;
      const rows=validationRows(model,heldout,mode);
      if(rows.some(r=>r.label===1)&&rows.some(r=>r.label===0))models[model.id][mode]=policy.frontier(rows);
    }
    console.log('Historical policy scores:',model.id);
  }
  fs.writeFileSync(path.join(root,'policy-validation.json'),JSON.stringify(artifact,null,2)+'\n');return artifact;
}
if(require.main===module){
  let current;try{current=JSON.parse(read('policy-validation.json'));}catch{}
  if(process.argv.includes('--if-stale')&&JSON.stringify(current?.sources)===JSON.stringify(signatures()))console.log('Historical policy scores are current.');
  else generate();
}
module.exports={validationRows,signatures,generate};
