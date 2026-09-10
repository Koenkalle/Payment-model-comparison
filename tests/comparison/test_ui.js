/* Deterministic DOM interaction harness; it is not a browser layout test. */
'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const model=JSON.parse(fs.readFileSync(__dirname+'/../../model-bundle.json','utf8'));
class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this.attrs={};this.listeners={};this.dataset={};this.style={};this.value='';this.textContent='';this.className='';this.checked=false;this.hidden=false;this.disabled=false;this.clientWidth=736;this.specialOptions={};}
  appendChild(e){this.children.push(e);return e;}replaceChildren(...e){this.children=e;}setAttribute(k,v){this.attrs[k]=String(v);}
  addEventListener(k,fn){this.listeners[k]=fn;}
  querySelector(s){
    if(s.startsWith('#'))return elements[s.slice(1)]||null;
    const option=s.match(/^option\[value="([^"]+)"\]$/);
    if(option)return this.specialOptions[option[1]]||(this.specialOptions[option[1]]=new Element('option'));
    return this.children.find(c=>c.tag===s)||null;
  }
  async fire(n){assert(this.listeners[n],'Missing handler '+n);await this.listeners[n]();await elements['fraud-memory-demo'].demo.whenIdle();}
}
const elements={},template=fs.readFileSync(__dirname+'/../../tools/comparison/template.html','utf8');
for(const m of template.matchAll(/<([a-z]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)){const e=new Element(m[1]);elements[m[2]]=e;const v=m[0].match(/\bvalue="([^"]*)"/);if(v)e.value=v[1];}
for(const[id,v]of Object.entries({'fd-scenario':'relay','fd-size':'small','fd-alpha':'0.02','fd-warmup':'128','fd-mode':'shadow','fd-training-mode':'unsupervised','fd-policy':'shared','fd-objective':'f1','fd-model-strategy':'shared','fd-model-alpha':'0.02','fd-model-warmup':'128','fd-model-tau':'10','fd-model-false-cost':'1','fd-model-missed-cost':'20','fd-model-objective':'f1','fd-delay':'24','fd-forward':'2','fd-seed':'42'}))elements[id].value=v;
for(const id of['fd-parts','fd-history','fd-metrics','fd-errors','fd-compare','fd-model-metrics'])elements[id].appendChild(new Element('tbody'));
elements['fd-model-data']=new Element('script');elements['fd-model-data'].textContent=JSON.stringify(model);
const document={getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
let scheduled;
const context={document,console,Intl,Math,JSON,Number,String,Array,Map,Set,WeakMap,Error,Date,Promise,setTimeout,clearTimeout,performance:require('perf_hooks').performance,ResizeObserver:class{constructor(f){this.f=f;}observe(){this.f();}},setInterval:fn=>(scheduled=fn,1),clearInterval:()=>scheduled=null};context.globalThis=context;vm.createContext(context);
const registry=require('../../tools/registry.json'),tool=registry.tools.find(t=>t.id==='comparison');
for(const file of require('../../shared/runtime/plugin-scripts')().concat(registry.shared_scripts,tool.scripts))vm.runInContext(fs.readFileSync(__dirname+'/../../'+file,'utf8'),context,{filename:file});
async function run(){
await elements['fraud-memory-demo'].demo.whenIdle();
const get=()=>elements['fraud-memory-demo'].demo.getSnapshot();const start=get();assert.strictEqual(start.accounts,32);assert(start.events>450);assert(start.tau>0);assert.strictEqual(start.comparison.length,9);assert.strictEqual(start.warmup,128);assert.strictEqual(start.trainingMode,'unsupervised');
assert(start.comparison.every(x=>JSON.stringify(x.payments)===JSON.stringify(start.comparison[0].payments)));
elements['fd-model'].value='statistics';await elements['fd-model'].fire('change');assert.strictEqual(get().model,'statistics');assert(elements['fd-memory-panels'].hidden);assert.strictEqual(get().count,start.count);assert.notStrictEqual(get().score,start.score);
elements['fd-model'].value='gru_attention';await elements['fd-model'].fire('change');assert.strictEqual(get().score,start.score);assert(!elements['fd-memory-panels'].hidden);
elements['fd-training-mode'].value='supervised';await elements['fd-training-mode'].fire('change');assert.strictEqual(get().trainingMode,'supervised');assert(Number.isFinite(get().score));assert(elements['fd-method'].textContent.includes('Supervised fraud head'));const supervisedScore=get().score;
elements['fd-training-mode'].value='unsupervised';await elements['fd-training-mode'].fire('change');assert.strictEqual(get().trainingMode,'unsupervised');assert(Number.isFinite(get().score));assert.notStrictEqual(get().score,supervisedScore);
elements['fd-model'].value='xgboost';await elements['fd-model'].fire('change');assert.strictEqual(get().model,'xgboost');assert(elements['fd-memory-panels'].hidden);assert(Number.isFinite(get().score));assert(elements['fd-training-source'].textContent.includes('flags are ignored'));
elements['fd-training-mode'].value='supervised';await elements['fd-training-mode'].fire('change');assert.strictEqual(get().trainingMode,'supervised');assert(Number.isFinite(get().score));assert(elements['fd-training-source'].textContent.includes('XGBoost checkpoint'));
elements['fd-training-mode'].value='unsupervised';await elements['fd-training-mode'].fire('change');assert.strictEqual(get().trainingMode,'unsupervised');assert(Number.isFinite(get().score));
for(const id of['dygformer','tami','dyg_tami','dyg_tami_gnn']){elements['fd-model'].value=id;await elements['fd-model'].fire('change');assert.strictEqual(get().model,id);assert(Number.isFinite(get().score));assert(elements['fd-memory-panels'].hidden);}
elements['fd-model'].value='gru_attention';await elements['fd-model'].fire('change');assert.strictEqual(get().score,start.score);
await elements['fd-next'].fire('click');let after=get();assert.strictEqual(after.count,start.count+1);assert.strictEqual(after.decisions.at(-1).score,start.score);assert.strictEqual(after.decisions.at(-1).tau,start.tau);
await elements['fd-back'].fire('click');assert.strictEqual(get().score,start.score);assert.strictEqual(get().tau,start.tau);
elements['fd-account'].value='5';await elements['fd-account'].fire('change');assert.strictEqual(get().selected,5);
elements['fd-truth'].checked=true;await elements['fd-truth'].fire('change');assert.strictEqual(elements['fd-evaluation'].hidden,false);
elements['fd-mode'].value='shadow';await elements['fd-mode'].fire('change');await elements['fd-next'].fire('click');assert(get().decisions.at(-1).settled);
elements['fd-alpha'].value='0.05';await elements['fd-alpha'].fire('change');assert(Number.isFinite(get().tau));
elements['fd-warmup'].value='64';await elements['fd-warmup'].fire('change');assert.strictEqual(get().warmup,64);assert(Number.isFinite(get().tau));assert.strictEqual(get().count,start.count+1);
await elements['fd-play'].fire('click');assert(scheduled);const before=get().count;scheduled();await elements['fraud-memory-demo'].demo.whenIdle();assert.strictEqual(get().count,before+1);await elements['fd-play'].fire('click');assert(!scheduled);
elements['fd-jump'].value='0';await elements['fd-jump'].fire('change');assert.strictEqual(get().count,0);assert.strictEqual(get().tau,null);assert(elements['fd-back'].disabled);
elements['fd-scenario'].value='benign';await elements['fd-scenario'].fire('change');assert(get().events>450);
elements['fd-size'].value='small';await elements['fd-size'].fire('change');assert.strictEqual(get().accounts,32);
await elements['fd-run'].fire('click');assert.strictEqual(get().count,get().events);assert(elements['fd-next'].disabled);
elements['fd-seed'].value='314';await elements['fd-seed'].fire('change');assert(get().count<get().events);
elements['fd-forward'].value='45';await elements['fd-forward'].fire('change');elements['fd-delay'].value='48';await elements['fd-delay'].fire('change');assert(get().score!==null);
elements['fd-graph-container'].clientWidth=320;elements['fd-chart-container'].clientWidth=320;await elements['fd-next'].fire('click');assert(elements['fd-graph'].attrs.viewBox.startsWith('0 0 320'));assert(elements['fd-chart'].attrs.viewBox.startsWith('0 0 320'));
elements['fd-mode'].value='enforce';await elements['fd-mode'].fire('change');assert.strictEqual(get().mode,'enforce');
elements['fd-mode'].value='shadow';await elements['fd-mode'].fire('change');const prePolicy=get();
elements['fd-policy'].value='individual';await elements['fd-policy'].fire('change');const individual=get();
assert.strictEqual(individual.policyScope,'individual');assert.strictEqual(individual.decisionPolicy,'shared');assert.strictEqual(individual.count,prePolicy.count);assert.strictEqual(individual.score,prePolicy.score);
assert(!elements['fd-individual-controls'].hidden);assert(elements['fd-auto-controls'].hidden);assert.strictEqual(elements['fd-model-strategy'].value,'shared');
elements['fd-model-strategy'].value='tuned';await elements['fd-model-strategy'].fire('change');const tuned=get();
assert.strictEqual(tuned.decisionPolicy,'tuned');assert.strictEqual(tuned.trainingMode,'unsupervised');assert(tuned.comparison.every(e=>Number.isFinite(e.tau)));assert(elements['fd-model-false-cost'].disabled===false);assert(elements['fd-model-missed-cost'].disabled===false);
assert(elements['fd-policy-summary'].textContent.includes('own error costs'));assert(elements['fd-alpha-label'].textContent.includes('Evaluation'));assert(tuned.decisions.every(d=>d.tau===tuned.tau&&d.decision!=='LEARNING'));
const tunedBlocks=tuned.policyFit.blocks;elements['fd-model-missed-cost'].value='100';await elements['fd-model-missed-cost'].fire('change');const highMissed=get();assert(highMissed.policyFit.blocks>=tunedBlocks);
elements['fd-alpha'].value='.01';await elements['fd-alpha'].fire('change');assert.strictEqual(get().tau,highMissed.tau);assert.strictEqual(get().rankingBudget,.01);
const beforePolicyStep=get();await elements['fd-next'].fire('click');assert.strictEqual(get().tau,beforePolicyStep.tau);await elements['fd-back'].fire('click');assert.strictEqual(get().score,beforePolicyStep.score);
elements['fd-model'].value='statistics';await elements['fd-model'].fire('change');assert.strictEqual(elements['fd-model-strategy'].value,'shared');
elements['fd-model-strategy'].value='manual';await elements['fd-model-strategy'].fire('change');elements['fd-model-tau'].value='5';await elements['fd-model-tau'].fire('change');const manual=get();
assert.strictEqual(manual.model,'statistics');assert.strictEqual(manual.decisionPolicy,'manual');assert.strictEqual(manual.tau,5);assert(manual.comparison.find(e=>e.id==='gru_attention').policy.includes('Cost tune'));
elements['fd-policy'].value='auto';await elements['fd-policy'].fire('change');const auto=get();
assert.strictEqual(auto.policyScope,'auto');assert.strictEqual(auto.decisionPolicy,'auto');assert(auto.comparison.every(e=>e.policy==='Auto F1'&&Number.isFinite(e.tau)));assert(!elements['fd-auto-controls'].hidden);assert(elements['fd-individual-controls'].hidden);assert(elements['fd-model-strategy'].disabled===false);
elements['fd-objective'].value='f2';await elements['fd-objective'].fire('change');const autoF2=get();assert(autoF2.comparison.every(e=>e.policy==='Auto F2'));assert.strictEqual(autoF2.decisionPolicy,'auto');
elements['fd-training-mode'].value='supervised';await elements['fd-training-mode'].fire('change');assert.strictEqual(get().trainingMode,'supervised');assert.strictEqual(get().decisionPolicy,'auto');assert(get().comparison.every(e=>Number.isFinite(e.tau)));
elements['fd-jump'].value='0';await elements['fd-jump'].fire('change');assert.strictEqual(get().count,0);assert(Number.isFinite(get().tau));
elements['fd-policy'].value='shared';await elements['fd-policy'].fire('change');assert.strictEqual(get().decisionPolicy,'shared');assert.strictEqual(get().tau,null);assert(!elements['fd-warmup'].disabled);assert(elements['fd-individual-controls'].hidden);
console.log('PASS UI: per-model cost tuning, fixed cutoffs, independent auto objectives, frozen τ, ranking budget and shared fallback.');
const demo=elements['fraud-memory-demo'].demo,inspectionCalls=demo.getPerformance().inferenceCalls;
elements['fd-model'].value='statistics';await elements['fd-model'].fire('change');
elements['fd-account'].value='4';await elements['fd-account'].fire('change');assert.strictEqual(demo.getPerformance().inferenceCalls,inspectionCalls);
elements['fd-size'].value='large';const outdated=elements['fd-size'].listeners.change();assert(get().busy);assert(!elements['fd-model'].disabled);assert(elements['fd-work-status'].textContent);
elements['fd-size'].value='small';elements['fd-seed'].value='928';const latest=elements['fd-seed'].listeners.change();
await Promise.all([outdated,latest]);await demo.whenIdle();assert(!get().busy);assert.strictEqual(get().accounts,32);assert.strictEqual(elements['fd-work-status'].textContent,'');
const expectedData=context.FraudScenarios.build(elements['fd-scenario'].value,'small',928,Number(elements['fd-delay'].value)*60,Number(elements['fd-forward'].value));
assert.strictEqual(get().events,expectedData.events.length);assert.strictEqual(get().count,expectedData.startIndex);assert(demo.getPerformance().comparisons<=3);
assert(!elements['fraud-memory-demo'].dataset.error);
console.log('PASS UI: inspection avoids inference, busy feedback is immediate, controls remain available and rapid changes display only the latest result.');
console.log('PASS UI: nine-model switching, shared histories, memory visibility, comparisons, pre-execution decision, rewind, account, truth, mode, alpha, play, jump, size, seed, timing, end, and 320px geometry.');
console.log('DOM interaction checks only; full browser CSS rendering is not verified.');

}
run().catch(error=>{console.error(error);process.exitCode=1;});
