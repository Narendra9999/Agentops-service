const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.join(__dirname,'../app/static');
const context={window:{},Intl,URLSearchParams,state:{workspace:'https://fevm-classic-stable-naren.cloud.databricks.com',workspaceId:'7474644082064164'}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'charts.js'),'utf8'),context);
const charts=context.window.GenieCharts;
const source=fs.readFileSync(path.join(root,'app.js'),'utf8');
vm.runInContext(source.slice(source.indexOf('function genieLink('),source.indexOf('function genieEvidence(')),context);
test('Genie deep link uses workspace monitoring route and selected message',()=>{
 const ids={space_id:'01f1d6efcecd1620b04cf487817675b7',conversation_id:'01f1dc8e33d9197490736ff7e9f372a1',message_id:'01f1dc8e33ef1de39831d8a5728abb78'};
 assert.equal(context.genieLink(ids),context.state.workspace+'/genie/rooms/'+ids.space_id+'/monitoring?o=7474644082064164&mc='+ids.conversation_id+'&m='+ids.message_id);
 assert.equal(context.genieLink({space_id:'javascript:alert(1)'}),null);
 assert.equal(context.genieLink({space_id:ids.space_id}),context.state.workspace+'/genie/rooms/'+ids.space_id+'?o=7474644082064164');
});
test('numeric zero and negatives render; missing, infinite and unsafe values are omitted',()=>{
 const table={columns:['scenario','cost'],rows:[['zero',0],['credit',-50],['empty',null],['blank',''],['infinite','Infinity'],['unsafe','9007199254740993']]};
 const result=charts.render(table,0,1,'bar');
 assert.match(result.svg,/zero · cost: 0/);assert.match(result.svg,/credit · cost: -50/);
 assert.match(result.svg,/width="0"/);assert.doesNotMatch(result.svg,/NaN|Infinity/);
 assert.match(result.note,/Showing 2 of 6/);
});
test('labels and column names cannot inject HTML',()=>{
 const table={columns:['<script>','<img>'],rows:[['<svg onload=alert(1)>',20]]};
 assert.doesNotMatch(charts.markup(table,0),/<script>|<img>/);
 assert.doesNotMatch(charts.render(table,0,1,'bar').svg,/<svg onload/);
 assert.match(charts.render(table,0,1,'bar').svg,/&lt;svg/);
});
test('line charts break at missing values and disclose preview limits',()=>{
 const table={columns:['scenario','cost'],rows:[['a',1],['b',null],['c',3]],truncated:true};
 const result=charts.render(table,0,1,'line');
 assert.match(result.svg,/<path[^>]+ d="M[^M]+M/);assert.match(result.note,/truncated/);
 assert.equal(charts.defaults({columns:['text'],rows:[['none']]}),null);
 const many={columns:['x','cost'],rows:Array.from({length:101},(_,i)=>['row',i])};
 assert.match(charts.render(many,0,1,'bar').note,/Showing 100 of 100/);
});

const costTable={columns:['scenario','core_monthly_usd','recurring_monthly_usd','one_time_usd','annual_usd','first_year_usd','pending_items'],rows:[['EXPECTED',62.07,72.08,0,864.92,864.92,0],['HIGH',124.15,134.15,0,1609.8,1609.8,0],['LOW',31.04,41.04,0,492.47,492.47,0]]};
test('scenario cost preset selects the three screenshot measures, excluding other numeric fields',()=>{
 const config=charts.defaults(costTable);
 assert.equal(config.kind,'grouped');assert.equal(config.x,0);
 assert.deepEqual(Array.from(config.ys),[2,4,5]);
 const markup=charts.markup(costTable,0);
 assert.match(markup,/Cost Estimate by Scenario/);
 assert.equal((markup.match(/ checked/g)||[]).length,3);
});
test('grouped cost chart shows nine bars with matching colors, legends, units and raw values',()=>{
 const result=charts.render(costTable,0,[2,4,5],'grouped');
 assert.equal((result.svg.match(/<rect /g)||[]).length,9);
 for(const color of ['#4c78a8','#f58518','#54a24b'])assert.match(result.svg,new RegExp('fill="'+color+'"'));
 for(const column of ['recurring_monthly_usd','annual_usd','first_year_usd'])assert.match(result.legend,new RegExp(column));
 assert.match(result.svg,/>USD<\/text>/);assert.match(result.svg,/>Scenario<\/text>/);
 assert.match(result.svg,/EXPECTED · recurring_monthly_usd: 72.08/);
 assert.match(result.svg,/HIGH · annual_usd: 1609.8/);
 const bars=[...result.svg.matchAll(/<rect data-series="([^"]+)" data-category="([^"]+)"[^>]+height="([^"]+)"/g)];
 const height=(series,category)=>Number(bars.find(b=>b[1]===series&&b[2]===category)[3]);
 assert.equal(height('annual_usd','HIGH'),height('first_year_usd','HIGH'));
 assert.ok(Math.abs(height('annual_usd','HIGH')/height('annual_usd','EXPECTED')-1609.8/864.92)<1e-10);
 assert.ok(result.svg.indexOf('>EXPECTED</text>')<result.svg.indexOf('>HIGH</text>'));
});
test('multiple line series retain independent gaps; empty selection is handled',()=>{
 const table={columns:['x','a','b'],rows:[['one',0,4],['two',null,3],['three',-2,null]]};
 const result=charts.render(table,0,[1,2],'line');
 assert.equal((result.svg.match(/<path /g)||[]).length,2);
 assert.match(result.svg,/<path data-series="a" d="M[^M]+M/);
 assert.doesNotMatch(result.svg,/NaN|Infinity/);
 assert.match(charts.render(table,0,[],'grouped').svg,/Select at least one/);
});

test('interactive plot preserves distinct rows and nulls without category aggregation',()=>{
 const table={columns:['scenario','annual_usd','first_year_usd'],rows:[['same',0,10],['same',null,20],['<b>other</b>',-3,30]]};
 const spec=charts.plotSpec(table,0,[1,2],'grouped');
 assert.equal(spec.layout.barmode,'group');
 assert.deepEqual(Array.from(spec.data[0].x),[0,1,2]);
 assert.deepEqual(Array.from(spec.data[0].y),[0,null,-3]);
 assert.equal(spec.layout.yaxis.title.text,'USD');
 assert.equal(spec.layout.xaxis.ticktext[2],'&lt;b&gt;other&lt;/b&gt;');
 assert.equal(spec.config.responsive,true);
 assert.equal(spec.layout.dragmode,'zoom');
});

test('agent-mode Genie link opens its chat conversation',()=>{
 const ids={space_id:'abc123',conversation_id:'def456',message_id:'aabb',mode:'agent'};
 assert.equal(context.genieLink(ids),context.state.workspace+'/genie/rooms/abc123/chats/def456?o=7474644082064164');
});
