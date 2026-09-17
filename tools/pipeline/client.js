/* Shared controls for the server-backed dataset and model preparation tools. */
(function(global){
  'use strict';
  const doc=global.document;
  const styles=doc.createElement('style');
  styles.textContent=`
    .pl-root{--pl-accent:light-dark(#127568,#71dbca);--pl-quiet:light-dark(#697578,#a5b2b7);--pl-line:light-dark(#dce5e2,#364246);--pl-panel:light-dark(#fff,#20272b);--pl-wash:light-dark(#f4f7f6,#182024);--pl-error:light-dark(#ae3743,#ffa0aa);max-width:1390px;margin:auto;padding:30px 32px 60px;color:var(--foreground);font-size:14px;line-height:1.5}
    .pl-root *{box-sizing:border-box}.pl-root [hidden]{display:none!important}.pl-root h1{font-size:34px;font-weight:650;letter-spacing:-1.1px;line-height:1.15;margin:8px 0 12px}.pl-root h2{font-size:19px;font-weight:620;letter-spacing:-.3px;margin:0}.pl-root h3{font-size:15px;font-weight:600;margin:0 0 7px}.pl-root p{margin:0}.pl-root .pl-eyebrow{color:var(--pl-accent);font-size:11px;letter-spacing:1.7px;font-weight:700;text-transform:uppercase}.pl-root .pl-muted,.pl-root .pl-help{color:var(--pl-quiet)}.pl-root .pl-small{font-size:12px}.pl-root .pl-help{font-size:11px;font-weight:400;line-height:1.5}.pl-root .pl-mono{font-family:var(--font-mono,monospace);font-size:11px;overflow-wrap:anywhere}
    .pl-root .pl-header{display:flex;justify-content:space-between;gap:20px;margin:26px 0;align-items:flex-start}.pl-root .pl-header p{max-width:740px}.pl-root .pl-steps{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:center;padding-bottom:20px;border-bottom:1px solid var(--pl-line)}.pl-root .pl-steps a{display:flex;align-items:center;gap:8px;color:var(--pl-quiet);text-decoration:none;font-size:12px}.pl-root .pl-steps a[aria-current="step"]{color:var(--pl-accent);font-weight:600}.pl-root .pl-step{display:grid;place-items:center;width:25px;height:25px;border-radius:50%;border:1px solid var(--pl-line);font-size:11px}.pl-root [aria-current="step"] .pl-step{border-color:var(--pl-accent);background:color-mix(in srgb,var(--pl-accent) 8%,transparent)}
    .pl-root .pl-grid{display:grid;grid-template-columns:minmax(300px,.85fr) minmax(0,1.4fr);gap:22px;align-items:start}.pl-root .pl-stack{display:grid;gap:18px;min-width:0}.pl-root .pl-panel{padding:22px;border:1px solid var(--pl-line);border-radius:14px;background:var(--pl-panel);min-width:0}.pl-root .pl-config{background:var(--pl-wash)}.pl-root .pl-line{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.pl-root .pl-actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.pl-root .pl-space{margin-top:18px}.pl-root .pl-gap{margin-top:10px}.pl-root .pl-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0}.pl-root .pl-field{display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:550;min-width:0}.pl-root .pl-fields .pl-wide{grid-column:1/-1}.pl-root :is(input,select,button){font:inherit}.pl-root :is(input:not([type="checkbox"]),select){height:40px;min-width:0;width:100%;border:1px solid var(--pl-line);border-radius:7px;padding:7px 10px;background:var(--pl-panel);color:var(--foreground);font-size:13px}.pl-root input[type="file"]{height:auto;font-size:12px}.pl-root input[type="checkbox"]{accent-color:var(--pl-accent);width:16px;height:16px}.pl-root :is(input,select,button,a,summary):focus-visible{outline:2px solid var(--pl-accent);outline-offset:3px}
    .pl-root button,.pl-root .pl-button{display:inline-flex;justify-content:center;align-items:center;gap:7px;border:1px solid var(--pl-line);border-radius:7px;padding:9px 14px;background:var(--pl-panel);color:var(--foreground);font-size:12px;font-weight:550;line-height:1.4;text-decoration:none;cursor:pointer}.pl-root :is(button,.pl-button):hover{background:var(--pl-wash)}.pl-root :is(button,.pl-button):disabled,.pl-root [aria-disabled="true"]{opacity:.45;pointer-events:none}.pl-root .pl-primary{background:var(--pl-accent);border-color:var(--pl-accent);color:light-dark(#fff,#112421)}.pl-root .pl-primary:hover{background:var(--pl-accent);filter:brightness(.94)}.pl-root fieldset{border:0;padding:0;margin:0;min-width:0}.pl-root fieldset:disabled{opacity:.65}.pl-root .pl-text-link{border:0;padding:0;color:var(--pl-accent);background:none;text-align:left;text-decoration:none}.pl-root .pl-text-link:hover{text-decoration:underline;background:none}
    .pl-root .pl-status{min-height:26px;margin:0 0 16px;color:var(--pl-quiet);font-size:12px}.pl-root .pl-error{padding:12px 16px;border-left:3px solid var(--pl-error);background:var(--pl-wash);color:var(--pl-error);margin-bottom:18px;font-size:13px;overflow-wrap:anywhere}.pl-root .pl-badge{display:inline-block;font-size:10px;font-weight:600;padding:3px 7px;border-radius:5px;background:var(--pl-wash);color:var(--pl-accent)}.pl-root .pl-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:20px 0}.pl-root .pl-stat{border:1px solid var(--pl-line);border-radius:10px;padding:13px 12px;min-width:0}.pl-root .pl-stat strong{display:block;font-size:24px;letter-spacing:-.5px;font-weight:620;margin:6px 0;font-variant-numeric:tabular-nums}.pl-root .pl-facts{display:grid;grid-template-columns:110px minmax(0,1fr);gap:9px 14px;font-size:12px;margin:18px 0 0}.pl-root dt{color:var(--pl-quiet)}.pl-root dd{margin:0;overflow-wrap:anywhere}.pl-root pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--pl-wash);padding:14px;border-radius:8px;font-size:11px;max-height:300px;overflow:auto}
    .pl-root .pl-table-wrap{overflow:auto;max-width:100%;scrollbar-width:thin}.pl-root table{border-collapse:collapse;width:100%;text-align:left;font-size:12px;font-variant-numeric:tabular-nums}.pl-root :is(th,td){padding:11px 10px;border-bottom:1px solid var(--pl-line);vertical-align:middle}.pl-root th{font-size:11px;font-weight:600;color:var(--pl-quiet);text-align:left;white-space:nowrap}.pl-root td{overflow-wrap:anywhere}.pl-root th:first-child,.pl-root td:first-child{padding-left:0}.pl-root tbody tr:last-child td{border-bottom:0}.pl-root tr[aria-selected="true"]{background:color-mix(in srgb,var(--pl-accent) 6%,transparent)}.pl-root .pl-sample td{white-space:nowrap;font-family:var(--font-mono,monospace);font-size:11px}.pl-root .pl-empty{border:1px dashed var(--pl-line);border-radius:10px;padding:30px 18px;text-align:center;color:var(--pl-quiet);font-size:13px}.pl-root details{margin-top:18px}.pl-root summary{cursor:pointer;font-size:12px;font-weight:550}.pl-root .pl-footer{margin-top:28px;border-top:1px solid var(--pl-line);padding-top:16px;color:var(--pl-quiet);font-size:11px}
    .pl-root .pl-split{display:flex;height:12px;border-radius:5px;overflow:hidden;background:var(--pl-line);margin:18px 0 10px}.pl-root .pl-split span{transition:width .2s;min-width:0}.pl-root .pl-train{background:var(--pl-accent)}.pl-root .pl-validation{background:light-dark(#599eb8,#8cbdd8)}.pl-root .pl-test{background:light-dark(#b6c9bd,#85998f)}.pl-root .pl-legend{display:flex;flex-wrap:wrap;gap:7px 15px;font-size:11px;color:var(--pl-quiet)}.pl-root .pl-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}.pl-root .pl-job{padding:16px 0;border-bottom:1px solid var(--pl-line)}.pl-root .pl-job:last-child{border-bottom:0;padding-bottom:0}.pl-root progress{width:100%;height:6px;accent-color:var(--pl-accent);margin-top:10px}.pl-root .pl-run-metrics{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--pl-quiet);margin-top:8px}.pl-root .pl-check{display:flex;gap:8px;align-items:center;font-size:12px}.pl-root .pl-note{padding:12px 14px;background:var(--pl-wash);border-radius:8px;font-size:12px;color:var(--pl-quiet)}
    @media(max-width:1080px){.pl-root{padding:24px 20px 40px}.pl-root .pl-grid{grid-template-columns:minmax(290px,.85fr) minmax(0,1.2fr)}.pl-root .pl-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:760px){.pl-root .pl-grid{grid-template-columns:1fr}.pl-root .pl-header{margin:22px 0}.pl-root .pl-steps{gap:9px 17px}.pl-root .pl-header h1{font-size:30px}}
    @media(max-width:440px){.pl-root{padding:20px 12px 32px}.pl-root .pl-panel{padding:17px}.pl-root .pl-steps{gap:8px}.pl-root .pl-steps a{font-size:11px;gap:5px}.pl-root .pl-step{width:21px;height:21px}.pl-root .pl-fields{gap:12px}.pl-root .pl-facts{grid-template-columns:88px minmax(0,1fr)}.pl-root .pl-header{flex-wrap:wrap}.pl-root .pl-stat strong{font-size:23px}}
  `;
  doc.head.appendChild(styles);
  function element(tag,parent,text,attrs={}){
    const node=doc.createElement(tag);
    if(text!==undefined&&text!==null)node.textContent=String(text);
    for(const [key,value] of Object.entries(attrs))if(value!==undefined&&value!==null)node.setAttribute(key,String(value));
    if(parent)parent.appendChild(node);
    return node;
  }
  async function request(path,body){
    if(!/^https?:$/.test(global.location.protocol))throw Error('Start the local app with python serve.py, then open this page at its HTTP address.');
    let response;
    try{response=await global.fetch('/api/pipeline'+path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
    catch(_){throw Error('The local server is unavailable. Start it with python serve.py and refresh.');}
    const payload=await response.json().catch(()=>null);
    if(!response.ok)throw Error(payload?.error?.message||payload?.error||payload?.message||'The request failed ('+response.status+'). Start the latest server with python serve.py.');
    if(!payload)throw Error('The server returned an unreadable response. Restart it with python serve.py.');
    return payload;
  }
  const number=(value,digits=0)=>Number.isFinite(Number(value))&&value!==null&&value!==''?Number(value).toLocaleString(undefined,{maximumFractionDigits:digits}):'—';
  const human=value=>String(value||'').replace(/_/g,' ').replace(/^./,char=>char.toUpperCase());
  function parameters(container,fields,prefix){
    container.replaceChildren();
    const schema=Array.isArray(fields)?fields:Object.entries(fields||{}).map(([name,field])=>({name,...field}));
    for(const field of schema){
      const label=element('label',container,null,{class:'pl-field'}),heading=element('span',label,field.label||human(field.name)),id=prefix+'-'+field.name;
      label.htmlFor=id;
      let input;
      if(field.options||field.type==='select'||field.type==='enum'){
        input=element('select',label,null,{id,name:field.name});
        const choices=(field.options||[]).map(choice=>choice!==null&&typeof choice==='object'?choice.value:choice);
        input.dataset.parameterOptions=JSON.stringify(choices);
        for(const choice of field.options||[]){const object=choice!==null&&typeof choice==='object',value=object?choice.value:choice;element('option',input,object?choice.label??value:value===null?'None':human(choice),{value:String(value)});}
      }else if(field.type==='boolean'){
        input=element('select',label,null,{id,name:field.name});element('option',input,'Yes',{value:'true'});element('option',input,'No',{value:'false'});
      }else{
        const numeric=['number','integer','float','int'].includes(field.type);
        input=element('input',label,null,{id,name:field.name,type:numeric?'number':'text',min:field.min,max:field.max,step:field.step??(['integer','int'].includes(field.type)?1:'any')});
      }
      input.value=field.default===null?(input.dataset.parameterOptions?'null':''):String(field.default??'');
      input.dataset.parameterType=field.type||'string';
      if(field.nullable)input.dataset.parameterNullable='true';
      input.required=field.required!==false&&field.default!==null;
      if(field.description||field.help)element('span',label,field.description||field.help,{class:'pl-help'});
      global.PaymentInfo.attach(label,global.PaymentInfo.parameter(field),{buttonParent:heading});
    }
    if(!schema.length)element('p',container,'This model uses its default configuration.',{class:'pl-muted pl-small'});
    return schema;
  }
  function featureList(parent,names,definitions=[]){
    parent.replaceChildren();
    const schema=Array.isArray(definitions)?definitions:Object.entries(definitions||{}).map(([id,feature])=>({id,...feature}));
    const catalog=new Map(schema.map(feature=>[feature.id||feature.name,feature]));
    if(!names?.length){parent.textContent='Feature information unavailable.';return;}
    for(const [index,name] of names.entries()){
      if(index)parent.append(doc.createTextNode(', '));
      const feature=catalog.get(name)||{id:name,label:human(name),info:{sections:[{title:'Saved definition',text:'The dataset response did not include this column’s saved definition. Any general explanation below must be checked against the source schema and saved feature recipe.'}]}};
      const label=element('span',parent,name,{class:'pl-feature-name','data-feature-id':name});
      global.PaymentInfo.attach(label,global.PaymentInfo.feature(feature));
    }
  }
  function parameterList(parent,values,fields=[]){
    parent.replaceChildren();parent.classList.add('pl-saved-parameters');
    const schema=Array.isArray(fields)?fields:Object.entries(fields||{}).map(([name,field])=>({name,...field})),catalog=new Map(schema.map(field=>[field.name,field]));
    for(const [name,value] of Object.entries(values||{})){
      const field=catalog.get(name)||{name,label:human(name),description:'A parameter stored with this model run.'};
      const shown=value===null?(field.default_label||'None'):typeof value==='object'?JSON.stringify(value):String(value);
      const row=element('div',parent,null,{'data-parameter-name':name});
      element('span',row,(field.label||human(name))+': ');
      element('span',row,shown,{class:'pl-mono'});
      const info=global.PaymentInfo.parameter(field);info.facts=[['Saved value',shown],...(info.facts||[])];
      global.PaymentInfo.attach(row,info);
    }
    if(!Object.keys(values||{}).length)parent.textContent='Default model configuration';
  }
  function values(container){
    const result={};
    for(const input of container.querySelectorAll('[data-parameter-type]')){
      if(!input.reportValidity())throw Error('Check the highlighted parameter.');
      const value=input.value.trim(),type=input.dataset.parameterType;
      if(value===''){if(input.required)throw Error(human(input.name)+' is required.');if(input.dataset.parameterNullable==='true')result[input.name]=null;continue;}
      result[input.name]=input.dataset.parameterOptions?JSON.parse(input.dataset.parameterOptions).find(option=>String(option)===value):['number','integer','float','int'].includes(type)?Number(value):type==='boolean'?value==='true':value;
    }
    return result;
  }
  function stat(parent,label,value,note){const card=element('div',parent,null,{class:'pl-stat'});element('div',card,label,{class:'pl-muted pl-small'});element('strong',card,value);if(note)element('div',card,note,{class:'pl-muted pl-small'});return card;}
  function facts(parent,entries){parent.replaceChildren();for(const [label,value] of entries){element('dt',parent,label);element('dd',parent,typeof value==='object'?JSON.stringify(value):value??'—');}}
  function labels(dataset){const known=dataset.known??dataset.labelcounts?.known??((dataset.fraud??0)+(dataset.legitimate??0));return {known,fraud:dataset.fraud??dataset.labelcounts?.fraud??0,unknown:dataset.unknown??Math.max(0,(dataset.rows||0)-known)};}
  function datasetStats(parent,dataset){
    parent.replaceChildren();const counts=labels(dataset);
    stat(parent,'Transactions',number(dataset.rows),dataset.kind==='generator'?'Generated and stored':'Fixed dataset');
    stat(parent,'Known outcomes',number(counts.known),number(counts.unknown)+' unknown');
    stat(parent,'Fraud labels',number(counts.fraud),counts.known?number(100*counts.fraud/counts.known,2)+'% of known outcomes':'No known outcomes');
    stat(parent,'Accounts',dataset.accounts?number(dataset.accounts):'Tabular',dataset.accounts?'Graph view available':'Numeric features available');
  }
  function date(value){if(!value)return '—';const parsed=new Date(value);return Number.isNaN(parsed.valueOf())?String(value):parsed.toLocaleString();}
  function link(page,params){return page+'?'+new URLSearchParams(Object.entries(params).filter(([,value])=>value!==null&&value!==undefined&&value!==''));}
  function remember(key,value){try{global.localStorage.setItem('payment-pipeline:'+key,value);}catch(_){}}
  function recall(key){try{return global.localStorage.getItem('payment-pipeline:'+key);}catch(_){return null;}}
  global.PaymentPipelineUI={element,request,number,human,parameters,parameterList,featureList,values,stat,facts,labels,datasetStats,date,link,remember,recall};
})(globalThis);
