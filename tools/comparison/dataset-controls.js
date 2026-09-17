/* The Python dataset module supplies browser-ready payments; models share the result. */
(function(global){
  'use strict';
  function mount(root,{importDataset,invalidate}){
    const $=id=>root.querySelector('#'+id),source=$('fd-benchmark-source'),button=$('fd-benchmark-load');
    let entries=[],catalog=null,controller=null,version=0;
    const selected=()=>entries.find(entry=>entry.id===source.value);
    function cancel(){version++;controller?.abort();controller=null;button.disabled=!selected()?.supported;}
    function changed(event){
      cancel();if(event!==false)invalidate();
      const entry=selected();
      $('fd-benchmark-upload').hidden=!entry?.upload||!entry.supported;
      $('fd-benchmark-currency').hidden=!entry?.requires_conversion||!entry.supported;
      $('fd-benchmark-help').textContent=entry?.description||'Choose a dataset.';
      $('fd-benchmark-start').value=String(entry?.selection?.start??'');
      $('fd-benchmark-stop').value=String(entry?.selection?.stop??'');
      $('fd-benchmark-factor').value='';$('fd-benchmark-file').value='';
      $('fd-benchmark-release').value='';
    }
    source.addEventListener('change',changed);
    async function load(){
      cancel();const revision=version,entry=selected();
      if(!entry?.supported)return;
      controller=new AbortController();const signal=controller.signal;
      button.disabled=true;
      try{
        await importDataset(async()=>{
          const request={id:entry.id},selection={};
          for(const key of ['start','stop']){
            const raw=$('fd-benchmark-'+key).value.trim();
            if(raw){const value=Number(raw);if(!Number.isFinite(value)||value<0)throw Error('Time bounds must be finite, nonnegative seconds.');selection[key]=value;}
          }
          if(selection.start!==undefined&&selection.stop!==undefined&&selection.start>=selection.stop)throw Error('The end time must follow the start time.');
          request.selection=selection;
          if(entry.requires_conversion){
            const raw=$('fd-benchmark-factor').value.trim(),factor=Number(raw);
            if(!raw||!Number.isFinite(factor)||factor<=0)throw Error('Enter a positive EUR conversion for the source amounts.');
            request.amount_to_eur=factor;
          }
          if(entry.upload){
            const file=$('fd-benchmark-file').files?.[0];
            if(!file)throw Error('Choose a source CSV first.');
            if(file.size>catalog.max_upload_bytes)throw Error('CSV uploads are limited to 6 MiB. Register larger local data with --dataset-config when starting serve.py.');
            request.name=file.name;
            const release=$('fd-benchmark-release').value.trim();if(release)request.release=release;
            request.csv=await file.text();
          }
          if(signal.aborted)throw Error('Dataset load cancelled.');
          const response=await global.fetch('/api/datasets/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request),signal});
          const payload=await response.json();
          if(!response.ok)throw Error(payload.error||'Dataset could not be loaded.');
          if(!payload.dataset)throw Error('The dataset service returned no payments.');
          return payload.dataset;
        },'Loading '+entry.label+'…');
      }finally{if(revision===version){controller=null;button.disabled=false;}}
    }
    button.addEventListener('click',load);
    // Editing a pending request invalidates it before its response can replace data.
    for(const id of ['fd-benchmark-file','fd-benchmark-release','fd-benchmark-factor','fd-benchmark-start','fd-benchmark-stop']){
      $(id).addEventListener('change',()=>{cancel();invalidate();});
    }
    const ready=(async()=>{
      if(!/^https?:$/.test(global.location?.protocol||'')||typeof global.fetch!=='function'){
        source.replaceChildren();const option=global.document.createElement('option');option.textContent='Local app required';source.appendChild(option);return;
      }
      try{
        const response=await global.fetch('/api/datasets');
        if(!response.ok)throw Error('Dataset service unavailable. Restart the local app with python serve.py.');
        catalog=await response.json();
        if(!Array.isArray(catalog.datasets)||!catalog.datasets.length)throw Error('The dataset catalog is empty.');
        entries=catalog.datasets;source.replaceChildren();
        for(const entry of entries){const option=global.document.createElement('option');option.value=entry.id;option.textContent=entry.label+(entry.supported?'':' · tabular only');source.appendChild(option);}
        source.value=entries[0].id;source.disabled=false;changed(false);
      }catch(error){$('fd-benchmark-help').textContent=error.message;source.disabled=true;button.disabled=true;}
    })();
    return {ready,cancel};
  }
  global.FraudDatasetControls={mount};
})(globalThis);
