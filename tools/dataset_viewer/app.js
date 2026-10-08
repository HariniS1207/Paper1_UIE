"use strict";
const $=s=>document.querySelector(s);
const state={all:[],filtered:[],selected:[],start:0,pageSize:16,candidates:false};
const labels={clear:"Relatively clearer / moderately degraded scene",cast:"Color-cast or hazy scene",strong:"More strongly degraded / deeper-looking scene"};

async function init(){
  try{
    const res=await fetch('/api/pairs'); if(!res.ok) throw new Error(`Server returned ${res.status}`);
    const data=await res.json(); state.all=data.pairs; state.filtered=state.all.slice();
    const missing=data.missingFromTrainA.length+data.missingFromTrainB.length;
    const status=$('#dataset-status');
    status.textContent=`${data.validCount.toLocaleString()} verified filename pairs available.`+(missing?` ${data.missingFromTrainA.length} missing from trainA; ${data.missingFromTrainB.length} missing from trainB.`:' All trainA/trainB filenames match.');
    if(missing) status.classList.add('warning');
    render();
  }catch(e){$('#dataset-status').textContent=`Could not load dataset: ${e.message}. Start the local server using the instructions in README.md.`;$('#dataset-status').classList.add('warning');}
}
function visible(){return state.filtered.slice(state.start,state.start+state.pageSize)}
function setStart(i){state.start=Math.max(0,Math.min(i,Math.max(0,state.filtered.length-state.pageSize)));render()}
function render(){
  const list=$('#pair-list');list.replaceChildren();
  const items=visible();$('#empty-state').hidden=state.filtered.length!==0;
  $('#browse-count').textContent=state.filtered.length?`Showing ${state.start+1}–${state.start+items.length} of ${state.filtered.length.toLocaleString()} matching pairs`:'';
  for(const filename of items){
    const row=document.createElement('article');row.className='pair-row'+(state.selected.includes(filename)?' is-selected':'');
    for(const [folder,label] of [['trainA','Degraded Input'],['trainB','Reference']]){
      const cell=document.createElement('div');cell.className='image-cell';
      const title=document.createElement('div');title.className='image-label';title.textContent=label;
      const img=document.createElement('img');img.className='sample-image';img.loading='lazy';img.alt=`${label}: ${filename}`;img.src=`/dataset/${folder}/${encodeURIComponent(filename)}`;
      img.addEventListener('error',()=>{img.alt=`Unable to load ${folder}/${filename}`;img.style.background='#fee';});
      const name=document.createElement('div');name.className='filename-line';name.textContent=`${folder}/${filename}`;
      cell.append(title,img,name);row.append(cell);
    }
    const actions=document.createElement('div');actions.className='row-actions';
    const status=document.createElement('span');status.textContent=state.selected.includes(filename)?`Selected · Pair ${state.selected.indexOf(filename)+1}`:'Verified matching filename in trainA and trainB';
    const btn=document.createElement('button');btn.className='button select-button';btn.textContent=state.selected.includes(filename)?'Remove selection':'Select pair';btn.disabled=!state.selected.includes(filename)&&state.selected.length===3;btn.addEventListener('click',()=>toggle(filename));
    actions.append(status,btn);row.append(actions);list.append(row);
  }
  const prev=state.start>0,next=state.start+state.pageSize<state.filtered.length;
  $('#previous').disabled=$('#previous-bottom').disabled=!prev;$('#next').disabled=$('#next-bottom').disabled=!next;
  renderSelection();
}
function toggle(name){if(state.selected.includes(name))state.selected=state.selected.filter(n=>n!==name);else if(state.selected.length<3)state.selected.push(name);render()}
function renderSelection(){
  const count=state.selected.length;$('#selection-count').textContent=`${count} / 3 selected`;$('#selection-count').classList.toggle('selection-max',count===3);
  $('#selection-hint').textContent=count===3?'Three verified pairs selected. The export is ready.':`Select ${3-count} more ${3-count===1?'pair':'pairs'} above to complete the figure preview.`;
  $('#download').disabled=$('#copy-names').disabled=count!==3;
  const preview=$('#selected-preview');preview.replaceChildren();
  const header=document.createElement('div');header.className='preview-head';for(const t of ['Degraded Input','Reference']){const d=document.createElement('div');d.textContent=t;header.append(d)}preview.append(header);
  for(let i=0;i<3;i++){
    const pairLabel=document.createElement('div');pairLabel.className='preview-pair-label';pairLabel.textContent=`Pair ${i+1}`;preview.append(pairLabel);
    const row=document.createElement('div');row.className='preview-row';
    for(const folder of ['trainA','trainB']){const cell=document.createElement('div');cell.className='preview-cell';if(state.selected[i]){const img=document.createElement('img');img.loading='lazy';img.alt=`Pair ${i+1}, ${folder==='trainA'?'Degraded Input':'Reference'}: ${state.selected[i]}`;img.src=`/dataset/${folder}/${encodeURIComponent(state.selected[i])}`;cell.append(img)}else{cell.classList.add('preview-placeholder');cell.textContent=`Pair ${i+1}`}row.append(cell)}preview.append(row);
  }
  $('#filenames').textContent=formatNames();
}
function formatNames(){return state.selected.map((n,i)=>`Pair ${i+1}:\ntrainA/${n}\ntrainB/${n}`).join('\n\n')}

// Estimate simple appearance descriptors from the original input pixels. These are only
// browsing heuristics: they do not classify quality or alter the images.
async function descriptors(names){return Promise.all(names.map(name=>new Promise(resolve=>{const im=new Image();im.onload=()=>{try{const c=document.createElement('canvas');c.width=48;c.height=48;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(im,0,0,48,48);const d=x.getImageData(0,0,48,48).data;let lum=0,lum2=0,sat=0,r=0,g=0,b=0,n=48*48;for(let i=0;i<d.length;i+=4){const rr=d[i]/255,gg=d[i+1]/255,bb=d[i+2]/255,l=.2126*rr+.7152*gg+.0722*bb;lum+=l;lum2+=l*l;sat+=(Math.max(rr,gg,bb)-Math.min(rr,gg,bb));r+=rr;g+=gg;b+=bb}lum/=n;resolve([lum,Math.sqrt(Math.max(0,lum2/n-lum*lum)),sat/n,(r-g)/n,(g-b)/n])}catch{resolve([0,0,0,0,0])}};im.onerror=()=>resolve([0,0,0,0,0]);im.src=`/dataset/trainA/${encodeURIComponent(name)}`})))}
async function chooseCandidates(){
  const btn=$('#candidates');btn.disabled=true;btn.textContent='Analyzing samples…';$('#candidate-note').hidden=false;$('#candidate-note').textContent='Comparing simple color and brightness patterns across original input images…';
  try{
    const pool=state.all.slice(0,Math.min(320,state.all.length)),features=await descriptors(pool);
    if(!pool.length)return;
    const mean=features.reduce((a,f)=>a.map((v,i)=>v+f[i]/features.length),[0,0,0,0,0]);
    const std=mean.map((_,i)=>Math.sqrt(features.reduce((s,f)=>s+(f[i]-mean[i])**2,0)/features.length)||1);
    const clear=features.map((f,i)=>({i,f,score:f[0]+f[1]})).sort((a,b)=>b.score-a.score)[0];
    const hazy=features.map((f,i)=>({i,f,d:Math.abs((f[0]-mean[0])/std[0])+Math.abs((f[2]-mean[2])/std[2])})).sort((a,b)=>a.d-b.d).find(x=>x.i!==clear.i)||clear;
    const picks=[clear,hazy];
    const third=features.map((f,i)=>({i,f,d:Math.min(...picks.map(p=>f.reduce((s,v,j)=>s+((v-p.f[j])/std[j])**2,0)))})).filter(x=>!picks.some(p=>p.i===x.i)).sort((a,b)=>b.d-a.d)[0];
    if(third)picks.push(third);
    state.selected=picks.slice(0,3).map(p=>pool[p.i]);
    state.candidates=true;
    $('#candidate-note').textContent=`Three visually varied suggestions selected from ${pool.length} samples using input-image brightness, contrast, colorfulness, and color-balance differences. These are browsing heuristics only; please judge the samples visually.`;
    render();
  }catch(e){$('#candidate-note').textContent=`Could not suggest candidates: ${e.message}`}
  finally{btn.disabled=false;btn.textContent='Show representative candidates'}
}
function fitImage(img,x,y,w,h){const ratio=Math.min(w/img.naturalWidth,h/img.naturalHeight);const iw=img.naturalWidth*ratio,ih=img.naturalHeight*ratio;return{x:x+(w-iw)/2,y:y+(h-ih)/2,w:iw,h:ih}}
async function downloadFigure(){
  if(state.selected.length!==3)return;
  const button=$('#download');button.disabled=true;button.textContent='Preparing PNG…';
  try{
    const scale=2, width=2400, margin=64, headerH=76, rowH=600, labelH=42, imageH=rowH-labelH;
    const layoutWidth=width/scale, layoutHeight=margin*2+headerH+rowH*3;
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=layoutHeight*scale;
    const ctx=canvas.getContext('2d');ctx.scale(scale,scale);const w=layoutWidth,h=layoutHeight;
    ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#4d5358';ctx.lineWidth=1;
    const innerW=w-margin*2,colW=innerW/2;
    ctx.font='600 18px Arial, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
    ['Degraded Input','Reference'].forEach((s,i)=>ctx.fillText(s,margin+colW*(i+.5),margin+headerH/2));
    ctx.beginPath();ctx.moveTo(margin,margin+headerH);ctx.lineTo(w-margin,margin+headerH);ctx.stroke();
    for(let r=0;r<3;r++){
      const top=margin+headerH+r*rowH;
      for(let c=0;c<2;c++){
        const x=margin+c*colW,folder=c===0?'trainA':'trainB',img=new Image();img.src=`/dataset/${folder}/${encodeURIComponent(state.selected[r])}`;
        await new Promise((resolve,reject)=>{if(img.complete&&img.naturalWidth)return resolve();img.onload=resolve;img.onerror=()=>reject(new Error(`Could not load ${folder}/${state.selected[r]}`))});
        const box=fitImage(img,x+12,top+labelH+12,colW-24,imageH-24);ctx.drawImage(img,box.x,box.y,box.w,box.h);
        ctx.strokeRect(x,top,colW,rowH);
      }
      ctx.fillStyle='#222';ctx.textAlign='left';ctx.textBaseline='middle';ctx.font='14px Arial, sans-serif';ctx.fillText(`Pair ${r+1}`,margin+12,top+labelH/2);
      ctx.beginPath();ctx.moveTo(margin,top+labelH);ctx.lineTo(w-margin,top+labelH);ctx.moveTo(margin+colW,top);ctx.lineTo(margin+colW,top+rowH);ctx.stroke();
    }
    const link=document.createElement('a');link.download='figure2_euvp_representative_pairs.png';link.href=canvas.toDataURL('image/png');link.click();
  }catch(e){alert(`Figure export failed: ${e.message}`)}finally{button.disabled=state.selected.length!==3;button.textContent='Download Figure PNG'}
}

$('#previous').addEventListener('click',()=>setStart(state.start-state.pageSize));$('#previous-bottom').addEventListener('click',()=>setStart(state.start-state.pageSize));
$('#next').addEventListener('click',()=>setStart(state.start+state.pageSize));$('#next-bottom').addEventListener('click',()=>setStart(state.start+state.pageSize));
$('#random').addEventListener('click',()=>{if(state.filtered.length)setStart(Math.floor(Math.random()*state.filtered.length))});
$('#page-size').addEventListener('change',e=>{state.pageSize=Number(e.target.value);state.start=0;render()});
$('#search').addEventListener('input',e=>{const q=e.target.value.trim().toLocaleLowerCase();state.filtered=state.all.filter(n=>n.toLocaleLowerCase().includes(q));state.start=0;render()});
$('#clear-selection').addEventListener('click',()=>{state.selected=[];render()});$('#candidates').addEventListener('click',chooseCandidates);$('#download').addEventListener('click',downloadFigure);
$('#copy-names').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(formatNames());$('#copy-names').textContent='Copied';setTimeout(()=>$('#copy-names').textContent='Copy filenames',1400)}catch{$('#filenames').parentElement.open=true;$('#filenames').focus()}});
init();
