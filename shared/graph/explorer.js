/* Dataset-neutral explorer controller. Transport, graph state and Canvas rendering
   are separate so another tool can mount this component against the same API. */
(function(global){
  'use strict';
  const U=global.PaymentPipelineUI,e=U.element,M=global.DatasetGraphModel;
  const EMPTY=()=>({nodes:[],edges:[]});
  class DatasetGraphExplorer{
    constructor(panel,{prefix='gx',closeId=prefix+'-close',onClose=()=>{}}={}){
      this.panel=panel;this.prefix=prefix;this.closeId=closeId;this.onClose=onClose;this.datasetId=null;this.dataset=null;this.summary=null;this.graph=EMPTY();this.selection=null;this.history=[];this.page=null;this.error=null;this.busy=false;this.pending=new Set();this.revision=0;this.controller=null;this.renderer=null;this.filters={};this.context={mode:'sample'};this.searchResults=[];this.searchRevision=0;this.tablePage=0;
      this.build();this.bind();
    }
    $(id){return this.panel.querySelector('#'+this.prefix+'-'+id);}
    build(){
      global.DatasetGraphView.mount(this.panel,{prefix:this.prefix,closeId:this.closeId});
      this.tableKind='node';
    }
    bind(){
      const click=(id,fn)=>this.$(id).addEventListener('click',fn);
      this.panel.querySelector('#'+this.closeId).addEventListener('click',()=>this.close());
      this.$('filters').addEventListener('submit',event=>{event.preventDefault();this.run(()=>this.load({reset:true}));});
      click('next',()=>this.run(()=>this.load({next:true})));click('reset',()=>{this.resetControls();this.run(()=>this.load({reset:true}));});
      click('undo',()=>this.undo());click('cancel',()=>this.cancel());click('expand',()=>this.expand());click('focus',()=>this.focus());
      click('fit',()=>this.renderer?.fit());click('zoom-in',()=>this.renderer?.zoomBy(1.3));click('zoom-out',()=>this.renderer?.zoomBy(1/1.3));
      this.$('layout').addEventListener('change',()=>this.renderer?.setLayout(this.$('layout').value));this.$('labels').addEventListener('change',()=>this.renderer?.setLabels(this.$('labels').checked));
      this.$('direction').addEventListener('change',()=>{this.filters={...this.filters,direction:this.$('direction').value};this.expansionOffsets=new Map();});
      this.$('search-form').addEventListener('submit',event=>{event.preventDefault();this.run(()=>this.search());});
      click('export',()=>this.export());
      for(const kind of ['node','edge'])click('tabs-'+kind,()=>{this.tableKind=kind;this.tablePage=0;this.renderTable();});
      click('table-prev',()=>{this.tablePage=Math.max(0,this.tablePage-1);this.renderTable();});click('table-next',()=>{this.tablePage++;this.renderTable();});
    }
    setDataset(dataset){
      this.cancel(false);this.dataset=dataset;this.datasetId=dataset?.id||null;this.summary=null;this.graph=EMPTY();this.selection=null;this.history=[];this.page=null;this.filters={};this.context={mode:'sample'};this.error=null;this.searchResults=[];this.searchRevision++;
      this.panel.hidden=true;this.renderer?.destroy();this.renderer=null;this.expansionOffsets=new Map();this.resetControls();this.$('search').value='';this.$('search-results').replaceChildren();this.showError(null);this.$('title').textContent=dataset?.name||'Dataset graph';
      for(const id of ['summary','legend','inspect','nodes','edges','table-count'])this.$(id).replaceChildren();
      this.$('edge-type').replaceChildren();e('option',this.$('edge-type'),'All types',{value:''});this.$('empty').hidden=true;this.message('');this.updateButtons();
    }
    resetControls(){for(const [id,value] of Object.entries({'node-limit':100,'edge-limit':300,direction:'both',label:'',start:'',stop:'','edge-type':''}))this.$(id).value=value;}
    async open(){
      if(!this.datasetId)return;if(this.busy)return this.whenIdle();this.panel.hidden=false;this.ensureRenderer();
      this.controller?.abort();this.controller=new AbortController();this.busy=true;this.updateButtons();
      const openingRevision=this.revision;
      return this.run(async()=>{
        const revision=this.revision;
        if(!this.summary){
          this.message('Preparing the graph index and reading dataset structure…');
          const summary=await this.request('');if(revision!==this.revision)return;
          if(!summary.supported)throw Error(summary.reason||'This dataset has no stable graph identities.');
          this.summary=summary;this.$('edge-type').replaceChildren();e('option',this.$('edge-type'),'All types',{value:''});
          for(const item of summary.edge_types)e('option',this.$('edge-type'),U.human(item.type)+' ('+U.number(item.count)+')',{value:item.type});
          this.$('node-limit').max=summary.limits.max_nodes;this.$('edge-limit').max=summary.limits.max_edges;
          this.$('start').placeholder=U.number(summary.time_range?.start,3);this.$('stop').placeholder=U.number(summary.time_range?.end,3)+' (last event)';
        }
        if(!this.page)await this.load({reset:true});else this.render();
      }).finally(()=>{if(openingRevision===this.revision){this.busy=false;this.updateButtons();}});
    }
    close(){this.cancel(false);this.panel.hidden=true;this.renderer?.destroy();this.renderer=null;this.onClose();}
    ensureRenderer(){if(!this.renderer)this.renderer=new global.DatasetGraphRenderer(this.$('canvas'),{onSelect:selection=>this.select(selection),onExpand:id=>{this.select({kind:'node',id});this.expand();}});}
    run(action){
      const revision=this.revision;
      const task=Promise.resolve().then(action).catch(cause=>{if(revision===this.revision&&cause.name!=='AbortError'){this.showError(cause.message||String(cause));this.message('The graph could not be updated. Adjust the controls or retry.');}}).finally(()=>{this.pending.delete(task);if(revision===this.revision)this.updateButtons();});
      this.pending.add(task);return task;
    }
    async whenIdle(){while(this.pending.size)await Promise.all([...this.pending]);}
    cancel(announce=true){this.controller?.abort();this.controller=null;this.revision++;this.searchRevision++;this.busy=false;this.updateButtons();if(announce)this.message('Loading canceled. The previous view is preserved.');}
    async request(suffix,body,signal){
      const response=await fetch('/api/pipeline/datasets/'+encodeURIComponent(this.datasetId)+'/graph'+suffix,{cache:'no-store',signal:signal||this.controller?.signal,...(body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})});
      const result=await response.json();if(!response.ok)throw Error(result?.error?.message||result?.error||'Graph request failed ('+response.status+').');return result;
    }
    readFilters(){
      if(!this.$('filters').reportValidity())throw Error('Use whole-number limits within the displayed bounds.');
      const filters={node_limit:Number(this.$('node-limit').value),edge_limit:Number(this.$('edge-limit').value),direction:this.$('direction').value};
      for(const name of ['start','stop','label'])if(this.$(name).value!=='')filters[name]=Number(this.$(name).value);
      if(filters.start!==undefined&&filters.stop!==undefined&&filters.start>=filters.stop)throw Error('From time must be earlier than Until time.');
      if(this.$('edge-type').value)filters.edge_type=this.$('edge-type').value;return filters;
    }
    save(){return {graph:this.graph,selection:this.selection,page:this.page,filters:{...this.filters},context:{...this.context},expansionOffsets:new Map(this.expansionOffsets||[])};}
    remember(){if(this.page){this.history.push(this.save());if(this.history.length>12)this.history.shift();}}
    async load({reset=false,next=false,nodeId=null,merge=false}={}){
      if(!this.summary)return;
      const filters=reset?this.readFilters():{...this.filters};
      const context=nodeId?{mode:'neighbors',node_id:nodeId}:{...this.context};if(reset)context.mode='sample';if(reset)delete context.node_id;
      if(merge&&this.graph.edges.length>=filters.edge_limit){this.message('Relationship limit reached. Increase the limit and apply, or focus on one entity.');return;}
      const expansion=merge?this.expansionOffsets?.get(nodeId):null;
      if(expansion&&!expansion.has_more){this.message('All matching connections for this entity have already been loaded.');return;}
      if(expansion?.has_more&&expansion.next_offset===null){this.message('The entity limit prevents further expansion. Increase the limit and apply, or choose Focus here.');return;}
      this.controller?.abort();this.controller=new AbortController();const revision=++this.revision;this.busy=true;this.showError(null);this.updateButtons();
      this.message(merge?'Loading this entity’s connections…':'Loading a bounded graph slice…');
      const body={...filters,...context,offset:next?(this.page?.next_offset??0):0};
      if(merge){body.existing_node_ids=this.graph.nodes.map(node=>node.id);body.edge_limit=Math.max(1,filters.edge_limit-this.graph.edges.length);body.offset=expansion?.next_offset||0;}
      try{
        const result=await this.request('/query',body);if(revision!==this.revision)return;
        this.remember();this.filters=filters;const before={nodes:this.graph.nodes.length,edges:this.graph.edges.length};
        this.graph=M.mergeGraph(merge?this.graph:EMPTY(),result,{nodes:filters.node_limit,edges:filters.edge_limit});
        if(merge){this.expansionOffsets||=new Map();this.expansionOffsets.set(nodeId,result.page);}
        else{this.context=context;this.page=result.page;this.expansionOffsets=new Map();this.selection=nodeId?{kind:'node',id:nodeId}:null;}
        this.tablePage=0;this.render();
        const capped=result.truncated||this.graph.nodes.length>=filters.node_limit||this.graph.edges.length>=filters.edge_limit;
        const added=merge?'Added '+(this.graph.nodes.length-before.nodes)+' entities and '+(this.graph.edges.length-before.edges)+' relationships. ':'Slice loaded. ';
        this.message(!this.graph.edges.length?(this.graph.nodes.length?'This entity has no matching relationships in the selected direction and time range.':'No matching relationships. Adjust the filters to explore another part of the dataset.'):added+(capped?'A display limit was reached. Explore the next slice, increase limits, or focus on an entity. ':'')+(result.page.has_more?'More relationships are available.':'This query has no more relationships.'));
      }catch(cause){if(revision===this.revision&&cause.name!=='AbortError'){this.showError(cause.message);this.message('The graph could not be updated. The previous view is preserved.');}}
      finally{if(revision===this.revision){this.busy=false;this.updateButtons();}}
    }
    expand(){if(this.selection?.kind==='node')return this.run(()=>this.load({nodeId:this.selection.id,merge:true}));}
    focus(){if(this.selection?.kind==='node')return this.run(()=>this.load({nodeId:this.selection.id}));}
    undo(){
      if(!this.history.length)return;this.cancel(false);const snapshot=this.history.pop();Object.assign(this,snapshot);this.syncControls();this.render();this.message('Restored the previous graph view.');
    }
    syncControls(){for(const name of ['start','stop','label','direction'])this.$(name).value=this.filters[name]??'';this.$('edge-type').value=this.filters.edge_type||'';this.$('node-limit').value=this.filters.node_limit;this.$('edge-limit').value=this.filters.edge_limit;}
    select(selection){this.selection=selection;this.renderer?.select(selection);this.renderInspector();this.updateButtons();}
    selectedNode(){return this.graph.nodes.find(node=>node.id===this.selection?.id)||this.searchResults.find(node=>node.id===this.selection?.id);}
    async search(){
      const revision=this.revision,searchRevision=++this.searchRevision,q=this.$('search').value.trim();
      if(!q){this.$('search-results').replaceChildren();return;}
      this.$('search-results').textContent='Searching dataset entities…';
      try{
        const result=await this.request('/search?'+new URLSearchParams({q,limit:20}));if(revision!==this.revision||searchRevision!==this.searchRevision)return;
        this.searchResults=result.nodes;this.$('search-results').replaceChildren();
        for(const node of result.nodes){const button=e('button',this.$('search-results'),node.label+' · '+U.human(node.type),{type:'button','data-node-id':node.id});button.addEventListener('click',()=>{this.select({kind:'node',id:node.id});this.message('Entity selected. Choose Focus here to explore its neighborhood.');});}
        if(!result.nodes.length)e('p',this.$('search-results'),'No matching entities.',{class:'pl-muted pl-small'});
        if(result.has_more)e('p',this.$('search-results'),'Showing the first 20 matches. Refine your search.',{class:'pl-muted pl-small'});
      }catch(cause){if(revision===this.revision&&searchRevision===this.searchRevision&&cause.name!=='AbortError'){this.$('search-results').textContent='Search failed. Try again.';throw cause;}}
    }
    render(){
      this.ensureRenderer();this.renderer.setGraph(this.graph,{layout:this.$('layout').value,selected:this.selection,labels:this.$('labels').checked});
      this.$('empty').hidden=this.graph.nodes.length>0;
      const stats=M.structure(this.graph),summary=this.$('summary');summary.replaceChildren();
      for(const [label,value] of [['Visible entities',U.number(this.graph.nodes.length)+' / '+U.number(this.summary.counts.nodes)],['Visible relationships',U.number(this.graph.edges.length)+' / '+U.number(this.summary.counts.edges)],['Components in view',stats.components],['Isolated in view',stats.isolated],['Self-loops in view',stats.selfLoops]]){const item=e('span',summary);e('strong',item,value);item.append(' '+label.toLowerCase());}
      this.$('legend').replaceChildren();
      for(const item of this.summary.node_types){const span=e('span',this.$('legend'),null,{class:'gx-key'});e('i',span,null,{class:'gx-dot'}).style.background=global.DatasetGraphRenderer.colorForType(item.type);span.append(U.human(item.type)+' · '+U.number(item.count));}
      for(const [label,color] of [['Fraud relationship','var(--pl-error)'],['Legitimate / unknown','var(--pl-quiet)']]){const span=e('span',this.$('legend'),null,{class:'gx-key'});e('i',span,null,{class:'gx-dot'}).style.background=color;span.append(label);}
      this.renderInspector();this.renderTable();this.updateButtons();
    }
    renderInspector(){
      const target=this.$('inspect');target.replaceChildren();const item=this.selection?.kind==='node'?this.selectedNode():this.graph.edges.find(edge=>edge.id===this.selection?.id);
      if(!item){e('h3',target,'Inspect the structure');e('p',target,'Select an entity or relationship in the graph or tables below.',{class:'pl-muted pl-small'});return;}
      e('div',target,this.selection.kind==='node'?'Entity':'Relationship',{class:'pl-eyebrow'});e('h3',target,this.selection.kind==='node'?item.label:item.id,{class:'pl-gap gx-inspector-title'});
      const facts=e('dl',target,null,{class:'pl-facts'});
      if(this.selection.kind==='node'){
        U.facts(facts,[['Type',U.human(item.type)],['Total degree',item.degree],['Incoming',item.in_degree],['Outgoing',item.out_degree],['ID',item.id]]);
        const visible=this.graph.edges.filter(edge=>edge.source===item.id||edge.target===item.id).length;e('p',target,U.number(visible)+' incident relationships in this view.',{class:'pl-small pl-muted'});
      }else U.facts(facts,[['Type',U.human(item.type)],['From',this.nodeLabel(item.source)],['To',this.nodeLabel(item.target)],['Time (seconds)',U.number(item.time,6)],['Outcome',item.label===1?'Fraud':item.label===0?'Legitimate':'Unknown'],['Event ID',item.id]]);
      e('h3',target,'Properties',{class:'pl-space'});e('pre',target,JSON.stringify(item.properties||{},null,2));
    }
    nodeLabel(id){return this.graph.nodes.find(node=>node.id===id)?.label||id;}
    renderTable(){
      const kind=this.tableKind,rows=kind==='node'?this.graph.nodes:this.graph.edges;
      this.tablePage=Math.min(this.tablePage,Math.max(0,Math.ceil(rows.length/50)-1));
      const start=this.tablePage*50,end=Math.min(start+50,rows.length);
      this.$('nodes').hidden=kind!=='node';this.$('edges').hidden=kind!=='edge';
      for(const type of ['node','edge'])this.$('tabs-'+type).setAttribute('aria-pressed',String(type===kind));
      const container=this.$(kind==='node'?'nodes':'edges');container.replaceChildren();
      const table=e('table',container,null,{'aria-label':kind==='node'?'Visible graph entities':'Visible graph relationships'}),head=e('tr',e('thead',table));
      for(const label of kind==='node'?['Entity','Type','Full degree','In / out']:['Relationship','From → to','Type','Time (s)','Outcome'])e('th',head,label,{scope:'col'});
      const body=e('tbody',table);
      for(const item of rows.slice(start,end)){
        const row=e('tr',body),cell=e('td',row),button=e('button',cell,kind==='node'?item.label:item.id,{type:'button',class:'pl-text-link','data-graph-id':item.id});button.addEventListener('click',()=>this.select({kind,id:item.id}));
        const values=kind==='node'?[U.human(item.type),U.number(item.degree),U.number(item.in_degree)+' / '+U.number(item.out_degree)]:[this.nodeLabel(item.source)+' → '+this.nodeLabel(item.target),U.human(item.type),U.number(item.time,3),item.label===1?'Fraud':item.label===0?'Legitimate':'Unknown'];
        for(const value of values)e('td',row,value);
      }
      this.$('table-count').textContent=rows.length?(start+1)+'–'+end+' of '+U.number(rows.length)+' visible':'No visible rows';this.$('table-prev').disabled=start===0;this.$('table-next').disabled=end>=rows.length;
    }
    updateButtons(){
      if(!this.$('next'))return;const node=this.selection?.kind==='node';
      for(const id of ['apply','reset','search-button'])this.$(id).disabled=this.busy;
      this.$('next').disabled=this.busy||!this.page?.has_more||this.page?.next_offset==null;this.$('undo').disabled=this.busy||!this.history.length;
      this.$('expand').disabled=this.busy||!node;this.$('focus').disabled=this.busy||!node;this.$('export').disabled=this.busy||!this.page;this.$('cancel').hidden=!this.busy;
      this.panel.setAttribute('aria-busy',String(this.busy));
    }
    showError(message){this.error=message||null;this.$('error').textContent=this.error||'';this.$('error').hidden=!this.error;}
    message(message){this.$('status').textContent=message;}
    export(){
      const result={schema:'dataset-graph-export/v1',dataset:{id:this.datasetId,name:this.dataset.name,fingerprint:this.dataset.fingerprint},exported_at:new Date().toISOString(),scope:'visible subset',filters:this.filters,query:this.context,totals:this.summary.counts,structure:M.structure(this.graph),...this.graph};
      const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),link=e('a',this.panel,null,{href:url,download:this.datasetId+'-graph.json'});link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    getSnapshot(){return JSON.parse(JSON.stringify({datasetId:this.datasetId,summary:this.summary,graph:this.graph,selection:this.selection,busy:this.busy,error:this.error,page:this.page,historyLength:this.history.length}));}
  }
  global.DatasetGraphExplorer=DatasetGraphExplorer;
})(globalThis);
