'use strict';
// Charts use query-result cells only. No model-generated HTML, JS or chart specs.
window.GenieCharts=(()=>{
 const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const colors=['#4c78a8','#f58518','#54a24b','#e45756','#b279a2','#72b7b2','#ff9da6','#9d755d'];
 const costMeasures=['recurring_monthly_usd','annual_usd','first_year_usd'];
 const number=value=>{
  if(value===null||value===undefined||String(value).trim()===''||!/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(String(value).trim()))return null;
  const n=Number(value);return Number.isFinite(n)&&(!Number.isInteger(n)||Number.isSafeInteger(n))?n:null;
 };
 const format=n=>new Intl.NumberFormat('en-US',{maximumSignificantDigits:6}).format(n);
 function defaults(table){
  if(!Array.isArray(table?.columns)||!Array.isArray(table?.rows))return null;
  const numeric=table.columns.map((_,i)=>i).filter(i=>table.rows.some(r=>number(r?.[i])!==null));
  if(!numeric.length||!table.rows.length)return null;
  const scenario=table.columns.indexOf('scenario'),cost=costMeasures.map(name=>table.columns.indexOf(name));
  const costPreset=scenario>=0&&cost.every(i=>numeric.includes(i));
  const category=table.columns.findIndex((_,i)=>!numeric.includes(i));
  return {x:costPreset?scenario:category<0?-1:category,y:numeric[0],ys:costPreset?cost:[numeric[0]],kind:costPreset?'grouped':'bar',costPreset,numeric};
 }
 function markup(table,index){
  const config=defaults(table);if(!config)return '';
  const options=[-1,...table.columns.map((_,i)=>i)].map(i=>'<option value="'+i+'"'+(i===config.x?' selected':'')+'>'+escape(i<0?'Row number':table.columns[i])+'</option>').join('');
  const measures=config.numeric.map(i=>'<label><input class="chart-measure" type="checkbox" value="'+i+'"'+(config.ys.includes(i)?' checked':'')+'><span>'+escape(table.columns[i])+'</span></label>').join('');
  return '<section class="genie-chart" data-table-index="'+index+'"><div class="chart-heading"><h4>'+ (config.costPreset?'Cost Estimate by Scenario':'Chart from Genie query result')+'</h4><button type="button" class="chart-download text-button" aria-label="Download chart as SVG">Download SVG ↓</button></div><div class="chart-controls"><label>Chart type<select class="chart-kind"><option value="grouped">Grouped columns</option><option value="bar">Horizontal bars</option><option value="line">Line</option></select></label><label>Category / X axis<select class="chart-x">'+options+'</select></label></div><details class="chart-measure-picker"><summary>Choose measures</summary><fieldset><legend>Measures / Y axis</legend>'+measures+'</fieldset></details><p class="chart-caption">Uses the displayed query rows without aggregation. Monthly and annual measures have different time periods.</p><div class="chart-canvas"></div><div class="chart-legend" aria-label="Chart legend"></div><p class="chart-row-note"></p></section>';
 }
 function render(table,x,y,kind){
  const ys=[...new Set(Array.isArray(y)?y:[y])].filter(i=>Number.isInteger(i)&&i>=0&&i<table.columns.length);
  if(!ys.length)return {svg:'<p>Select at least one measure to draw the chart.</p>',legend:'',note:''};
  const rows=table.rows.slice(0,100),label=(r,i)=>x<0?String(i+1):String(r[x]??'NULL');
  const series=ys.map((column,j)=>({column,name:table.columns[column],color:colors[j%colors.length],points:rows.map((r,i)=>({i,label:label(r,i),raw:r[column],value:number(r[column])})).filter(p=>p.value!==null)}));
  const points=series.flatMap(s=>s.points);
  if(!points.length)return {svg:'<p>No numeric values are available for these measures.</p>',legend:'',note:''};
  const minimum=Math.min(0,...points.map(p=>p.value)),maximum=Math.max(0,...points.map(p=>p.value));
  const axis=kind!=='bar';
  // Round vertical-axis limits to readable ticks while retaining a zero baseline.
  const rough=(maximum-minimum||1)/4,magnitude=10**Math.floor(Math.log10(rough));
  const step=[1,2,2.5,5,10].map(v=>v*magnitude).find(v=>v>=rough)||magnitude*10;
  const low=axis&&minimum<0?minimum*1.08:minimum,high=axis&&maximum>0?maximum*1.08:maximum;
  const range=high-low||1,W=900,pad=30;
  let drawing='',H;
  const tip=(s,p)=>'<title>'+escape(p.label+' · '+s.name+': '+p.raw)+'</title>';
  const text=(x,y,value,attrs='')=>'<text x="'+x+'" y="'+y+'" '+attrs+'>'+escape(value)+'</text>';
  if(kind==='bar'){
   const rowHeight=Math.max(34,series.length*20+12);H=Math.max(140,rows.length*rowHeight+60);
   const left=300,right=110,scale=v=>left+(v-low)/range*(W-left-right),zero=scale(0);
   drawing+='<line x1="'+zero+'" x2="'+zero+'" y1="20" y2="'+(H-30)+'" stroke="#94a3b8"/>';
   rows.forEach((r,i)=>{const name=label(r,i);drawing+=text(200,40+i*rowHeight,name.length>28?name.slice(0,27)+'…':name,'text-anchor="end"')});
   series.forEach((s,j)=>s.points.forEach(p=>{const cy=35+p.i*rowHeight+j*20,end=scale(p.value);drawing+='<rect data-series="'+escape(s.name)+'" x="'+Math.min(zero,end)+'" y="'+(cy-10)+'" width="'+Math.abs(end-zero)+'" height="18" fill="'+s.color+'">'+tip(s,p)+'</rect>'+text(end+(p.value<0?-7:7),cy+4,format(p.value),'text-anchor="'+(p.value<0?'end':'start')+'"')}));
  }else{
   H=440;const left=85,right=35,top=20,bottom=340,width=W-left-right;
   const scaleY=v=>bottom-(v-low)/range*(bottom-top),zero=scaleY(0);
   const scaleX=i=>kind==='grouped'?left+(i+.5)*width/rows.length:left+20+i/Math.max(1,rows.length-1)*(width-40);
   for(let i=0;i<=12;i++){const value=low+step*i;if(value>high+step*.001)break;const cy=scaleY(value);drawing+='<line x1="'+left+'" x2="'+(W-right)+'" y1="'+cy+'" y2="'+cy+'" stroke="#e2e8f0"/>'+text(left-8,cy+5,format(value),'text-anchor="end"');}
   drawing+='<line x1="'+left+'" x2="'+(W-right)+'" y1="'+zero+'" y2="'+zero+'" stroke="#9ca3af"/>';
   if(kind==='grouped'){
    const groupWidth=width/rows.length*.8,barWidth=groupWidth/series.length;
    series.forEach((s,j)=>s.points.forEach(p=>{const end=scaleY(p.value),bx=scaleX(p.i)-groupWidth/2+j*barWidth;drawing+='<rect data-series="'+escape(s.name)+'" data-category="'+escape(p.label)+'" x="'+bx+'" y="'+Math.min(zero,end)+'" width="'+barWidth+'" height="'+Math.abs(end-zero)+'" fill="'+s.color+'">'+tip(s,p)+'</rect>'}));
   }else{
    series.forEach(s=>{let path='',previous=-2;s.points.forEach(p=>{path+=(p.i===previous+1?'L':'M')+scaleX(p.i)+','+scaleY(p.value)+' ';previous=p.i});drawing+='<path data-series="'+escape(s.name)+'" d="'+path+'" fill="none" stroke="'+s.color+'" stroke-width="2.5"/>';s.points.forEach(p=>{drawing+='<circle cx="'+scaleX(p.i)+'" cy="'+scaleY(p.value)+'" r="4" fill="'+s.color+'">'+tip(s,p)+'</circle>'})});
   }
   rows.forEach((r,i)=>{if(i%Math.max(1,Math.ceil(rows.length/8))===0){const name=label(r,i);drawing+=text(scaleX(i),365,name.length>22?name.slice(0,21)+'…':name,'text-anchor="middle"')}});
   const unit=ys.every(i=>/_usd$/i.test(table.columns[i]))?'USD':'Value';
   drawing+='<text transform="translate(20,180) rotate(-90)" text-anchor="middle" class="axis-title">'+unit+'</text>'+text((left+W-right)/2,410,x<0?'Row number':table.columns[x]==='scenario'?'Scenario':table.columns[x],'text-anchor="middle" class="axis-title"');
  }
  const legend=series.map(s=>'<span><i style="background:'+s.color+'"></i>'+escape(s.name)+'</span>').join('');
  const validRows=new Set(points.map(p=>p.i)).size,missing=rows.length*series.length-points.length;
  return {svg:'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="'+H+'" viewBox="0 0 900 '+H+'" role="img" aria-label="'+escape(series.map(s=>s.name).join(', ')+' by '+(x<0?'row number':table.columns[x]))+'"><style>text{font:16px system-ui,sans-serif;fill:#17212b}.axis-title{font-size:20px;font-weight:600}</style>'+drawing+'</svg>',legend,note:'Showing '+validRows+' of '+rows.length+' preview rows. '+(missing?'Missing or nonnumeric values are omitted, not treated as zero. ':'')+(table.rows.length>100?'Only the first 100 rows are charted. ':'')+(table.truncated?'The query preview is truncated; open Genie for the full result.':'')};
 }
 function plotSpec(table,x,y,kind){
  const ys=[...new Set(Array.isArray(y)?y:[y])].filter(i=>Number.isInteger(i)&&i>=0&&i<table.columns.length);
  const rows=table.rows.slice(0,100),categories=rows.map((r,i)=>x<0?String(i+1):String(r[x]??'NULL'));
  // Positions identify rows independently: repeated category labels must not aggregate.
  const positions=rows.map((_,i)=>i);
  const data=ys.map((column,j)=>{
   const values=rows.map(r=>number(r[column])),labels=categories.map(escape),name=escape(table.columns[column]);
   const base={name,customdata:labels,marker:{color:colors[j%colors.length]},hovertemplate:'%{customdata}<br>'+name+': %{'+(kind==='bar'?'x':'y')+':,.2f}<extra></extra>'};
   return kind==='line'?{...base,type:'scatter',mode:'lines+markers',x:positions,y:values,connectgaps:false,line:{color:colors[j%colors.length]},marker:{...base.marker,size:7}}:
    {...base,type:'bar',orientation:kind==='bar'?'h':'v',x:kind==='bar'?values:positions,y:kind==='bar'?positions:values};
  });
  const categoryAxis={title:{text:x<0?'Row number':table.columns[x]==='scenario'?'Scenario':escape(table.columns[x])},tickmode:'array',tickvals:positions,ticktext:categories.map(escape),automargin:true,range:[-.6,Math.max(.6,rows.length-.4)]};
  const valueAxis={title:{text:ys.length&&ys.every(i=>/_usd$/i.test(table.columns[i]))?'USD':'Value'},rangemode:'tozero',nticks:5,separatethousands:true,automargin:true,gridcolor:'#e2e8f0',zerolinecolor:'#9ca3af'};
  return {data,layout:{height:kind==='bar'?Math.min(700,Math.max(400,rows.length*ys.length*23+140)):430,autosize:true,barmode:'group',bargap:.2,bargroupgap:0,
   margin:{l:70,r:25,t:20,b:100},paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',font:{family:'system-ui, sans-serif',color:'#17212b',size:13},
   xaxis:kind==='bar'?valueAxis:categoryAxis,yaxis:kind==='bar'?categoryAxis:valueAxis,
   legend:{orientation:'h',x:0,y:-.25,xanchor:'left',yanchor:'top'},hovermode:'closest',dragmode:'zoom',
   annotations:ys.length?[]:[{text:'Select at least one measure',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]},
   config:{responsive:true,displaylogo:false,scrollZoom:false,displayModeBar:true,toImageButtonOptions:{format:'png',filename:'genie-query-chart',scale:2}}};
 }
 function mount(parent,tables){
  parent.querySelectorAll('.genie-chart').forEach(block=>{
   const table=tables[Number(block.dataset.tableIndex)],config=defaults(table),x=block.querySelector('.chart-x'),measures=[...block.querySelectorAll('.chart-measure')],kind=block.querySelector('.chart-kind');
   const canvas=block.querySelector('.chart-canvas'),legend=block.querySelector('.chart-legend');
   let sequence=0;
   const update=()=>{
    const current=++sequence,selected=measures.filter(c=>c.checked).map(c=>Number(c.value));
    const fallback=()=>{try{const result=render(table,Number(x.value),selected,kind.value);canvas.classList.remove('interactive-chart');canvas.innerHTML=result.svg;legend.innerHTML=result.legend||'';block.querySelector('.chart-row-note').textContent=result.note;}catch{canvas.textContent='Chart unavailable. The query table and Genie link remain available.';legend.replaceChildren()}};
    if(!window.Plotly){fallback();return}
    const spec=plotSpec(table,Number(x.value),selected,kind.value);
    canvas.classList.add('interactive-chart');legend.replaceChildren();
    block.querySelector('.chart-row-note').textContent='Showing '+Math.min(100,table.rows.length)+' query rows. Hover for values; drag to zoom; use the toolbar to pan or reset. Click a legend entry to toggle its series.'+(table.truncated?' Preview truncated; open Genie for the full result.':'');
    Promise.resolve().then(()=>{if(canvas.isConnected)return window.Plotly.react(canvas,spec.data,spec.layout,spec.config)}).catch(()=>{if(sequence===current)fallback()});
   };
   // Explicit cost-scenario layout takes precedence over the generic chart preference.
   kind.value=config.costPreset?'grouped':window.Conversations?.chartType||config.kind;
   [x,...measures,kind].forEach(control=>control.addEventListener('change',update));update();
   block.querySelector('.chart-download').onclick=()=>{
    if(window.Plotly&&canvas.classList.contains('interactive-chart')){window.Plotly.downloadImage(canvas,{format:'svg',filename:'genie-query-chart',width:1000,height:600});return}
    const svg=block.querySelector('svg');if(!svg)return;
    const copy=svg.cloneNode(true),ns='http://www.w3.org/2000/svg';
    const oldHeight=Number(copy.getAttribute('height')),extra=55+seriesLegendHeight();
    copy.setAttribute('height',oldHeight+extra);copy.setAttribute('viewBox','0 -40 900 '+(oldHeight+extra));
    const add=(value,px,py,size=14)=>{const t=document.createElementNS(ns,'text');t.setAttribute('x',px);t.setAttribute('y',py);t.setAttribute('style','font:'+size+'px system-ui;fill:#17212b');t.textContent=value;copy.append(t)};
    add(block.querySelector('h4').textContent,25,-10,22);
    measures.filter(c=>c.checked).forEach((c,j)=>{const rect=document.createElementNS(ns,'rect'),lx=20+(j%3)*290,ly=oldHeight+Math.floor(j/3)*30;rect.setAttribute('x',lx);rect.setAttribute('y',ly);rect.setAttribute('width',14);rect.setAttribute('height',14);rect.setAttribute('fill',colors[j%colors.length]);copy.append(rect);add(table.columns[Number(c.value)],lx+20,ly+12)});
    const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='genie-query-chart.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
   };
   function seriesLegendHeight(){return Math.ceil(measures.filter(c=>c.checked).length/3)*30}
  });
 }
 function cleanup(parent){if(window.Plotly)parent.querySelectorAll('.js-plotly-plot').forEach(graph=>window.Plotly.purge(graph))}
 return {markup,mount,render,defaults,plotSpec,cleanup};
})();
