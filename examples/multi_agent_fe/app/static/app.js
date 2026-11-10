'use strict';
const el=id=>document.getElementById(id);
const state={history:[],busy:false,workspace:null,workspaceId:null,experiment:null,started:0,timer:null,controller:null,events:[]};
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// A deliberately small renderer: model HTML is always text; only explicit HTTPS
// links are clickable. No model string is inserted as executable HTML.
function inline(text){
 const tokens=[];
 let safe=escapeHTML(text).replace(/`([^`]+)`/g,(_,v)=>{tokens.push('<code>'+v+'</code>');return '\u0000'+(tokens.length-1)+'\u0000'});
 safe=safe.replace(/\[([^\]]+)\]\((https:\/\/[^\s)]+)\)/g,(_,label,url)=>{tokens.push('<a href="'+url+'" target="_blank" rel="noopener noreferrer">'+label+'</a>');return '\u0000'+(tokens.length-1)+'\u0000'});
 safe=safe.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
 return safe.replace(/\u0000(\d+)\u0000/g,(_,n)=>tokens[Number(n)]);
}
function markdown(text){
 const lines=String(text).replace(/\r/g,'').split('\n');let out='',i=0;
 const row=s=>s.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(v=>v.trim());
 while(i<lines.length){let line=lines[i];
  if(/^```/.test(line)){const code=[];i++;while(i<lines.length&&!/^```/.test(lines[i]))code.push(lines[i++]);i++;out+='<pre><code>'+escapeHTML(code.join('\n'))+'</code></pre>';continue}
  if(!line.trim()||/^\s*---+\s*$/.test(line)){i++;continue}
  if(i+1<lines.length&&line.includes('|')&&/^\s*\|?\s*:?-{3,}/.test(lines[i+1])){const head=row(line);i+=2;out+='<div class="table-wrap"><table><thead><tr>'+head.map(c=>'<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>';while(i<lines.length&&lines[i].includes('|'))out+='<tr>'+row(lines[i++]).map(c=>'<td>'+inline(c)+'</td>').join('')+'</tr>';out+='</tbody></table></div>';continue}
  const h=line.match(/^(#{1,6})\s+(.+)/);if(h){const level=Math.min(h[1].length+1,4);out+='<h'+level+'>'+inline(h[2])+'</h'+level+'>';i++;continue}
  if(/^\s*([-*]|\d+\.)\s+/.test(line)){const ordered=/^\s*\d+\./.test(line),tag=ordered?'ol':'ul';out+='<'+tag+'>';while(i<lines.length&&/^\s*([-*]|\d+\.)\s+/.test(lines[i]))out+='<li>'+inline(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/,''))+'</li>';out+='</'+tag+'>';continue}
  out+='<p>'+inline(line)+'</p>';i++;
 }
 return out;
}
function traceURL(id){if(!state.workspace||!state.experiment||!/^tr-[a-f0-9]+$/i.test(id||''))return null;return state.workspace.replace(/\/$/,'')+'/ml/experiments/'+encodeURIComponent(state.experiment)+'/traces?selectedTraceId='+encodeURIComponent(id)}
const agentNames={supervisor:'Supervisor',cost_estimation:'Cost intelligence · Genie',platform_docs:'Platform knowledge'};
function activity(phase,seconds=0){
 if(phase==='accepted')state.events=[{title:'Request received',message:'Connecting to the supervisor.'}];
 if(['done','failed','stopped'].includes(phase))state.events.push({title:phase==='done'?'Response ready':phase==='stopped'?'Stopped waiting':'Execution interrupted',message:phase==='done'?'The final answer is available in the conversation.':'Inspect the execution trace and last reported tool status.'});
 el('timeline').innerHTML=state.events.map((step,i)=>'<li class="'+(i<state.events.length-1||phase==='done'?'complete':['failed','stopped'].includes(phase)?'failed':'active')+'"><span class="step-dot"></span><div><strong>'+escapeHTML(step.title)+'</strong><p>'+escapeHTML(step.message)+'</p>'+(step.tool?'<code>'+escapeHTML(step.tool)+'</code>':'')+'</div></li>').join('');
 el('activity-badge').textContent=phase==='done'?'Complete':phase==='stopped'?'Stopped':phase==='failed'?'Interrupted':'Running';
 el('activity-badge').className='badge '+(phase==='done'?'done':['failed','stopped'].includes(phase)?'failed':'running');
 el('duration').hidden=false;el('duration').textContent=(phase==='done'?'Completed in ':'Elapsed time: ')+seconds+'s';
 if(phase==='done')el('next-step').textContent='Completed';
 if(['failed','stopped'].includes(phase))el('next-step').textContent='Verify the last operation before retrying';
}
function executionActivity(data,seconds){
 const name=agentNames[data.agent]||data.agent;
 const labels={genie_status:'Cost intelligence · Genie status',request_started:'Preparing the agent team',supervisor_started:'Supervisor · selecting next step',plan:'Plan → '+(agentNames[data.next_agent]||data.next_agent),agent_started:name+' · started',agent_completed:name+' · completed',tool_started:name+' · calling tool',tool_completed:name+' · tool returned',tool_failed:name+' · tool failed',synthesis_completed:'Supervisor · response prepared',receipt_replayed:'Existing receipt'};
 state.events.push({title:labels[data.kind]||name,message:data.message||'',tool:data.tool});
 const next={genie_status:'Await Genie query completion',request_started:'Supervisor reviews your request',supervisor_started:'Awaiting the routing decision',plan:'Call '+(agentNames[data.next_agent]||data.next_agent),agent_started:'Specialist selects its tool or response',tool_started:'Await the tool result',tool_completed:'Specialist reviews the tool result',tool_failed:'Check the failed operation',agent_completed:'Supervisor reviews the result and chooses the next step',synthesis_completed:'Deliver the final answer',receipt_replayed:'Return the original receipt without another write'};
 el('next-step').textContent=next[data.kind]||'Continue the selected operation';
 el('plan-card').hidden=false;activity('running',seconds);
}
function showUser(text){const d=document.createElement('article');d.className='turn user';const b=document.createElement('div');b.className='user-bubble';b.textContent=text;d.append(b);el('chat').append(d)}
function genieLink(result){
 if(!state.workspace||!result||!/^[a-f0-9]+$/i.test(result.space_id||''))return null;
 const base=state.workspace+'/genie/rooms/'+encodeURIComponent(result.space_id);
 const params=new URLSearchParams();
 if(/^\d+$/.test(state.workspaceId||''))params.set('o',state.workspaceId);
 if(/^[a-f0-9]+$/i.test(result.conversation_id||'')){
  if(result.mode==='agent')return base+'/chats/'+encodeURIComponent(result.conversation_id)+(params.size?'?'+params.toString():'');
  params.set('mc',result.conversation_id);
  if(/^[a-f0-9]+$/i.test(result.message_id||''))params.set('m',result.message_id);
  return base+'/monitoring?'+params.toString();
 }
 const url=base+(params.size?'?'+params.toString():'');
 return url;
}
function genieEvidence(result){
 if(!result)return '';
 let html='';
 for(const [index,table] of (result.tables||[]).slice(0,10).entries()){
  const chart=window.GenieCharts?.markup(table,index)||'';
  html+=chart+'<details class="genie-evidence"'+(chart?'':' open')+'><summary>'+escapeHTML(table.title||'Genie query result')+'</summary><div class="table-wrap"><table><thead><tr>'+(table.columns||[]).slice(0,30).map(c=>'<th>'+escapeHTML(c)+'</th>').join('')+'</tr></thead><tbody>';
  html+=(table.rows||[]).slice(0,100).map(row=>'<tr>'+row.slice(0,30).map(v=>'<td>'+escapeHTML(v===null?'NULL':v)+'</td>').join('')+'</tr>').join('');
  html+='</tbody></table></div>'+(table.truncated?'<p>Showing a limited result preview. Open Genie for the full result.</p>':'')+'</details>';
 }
 const url=genieLink(result);
 if(url)html+='<p><a href="'+escapeHTML(url)+'" target="_blank" rel="noopener noreferrer">Open Genie conversation'+(result.has_visualization?' and native visualization':'')+' ↗</a></p>';
 return html;
}
function showAnswer(data,seconds){
 const answer=(data.output||[]).flatMap(x=>x.content||[]).filter(x=>typeof x.text==='string').map(x=>x.text).join('\n');
 if(!answer.trim())throw Error('The agent returned no readable answer. Inspect its execution trace before retrying.');
 const url=traceURL(data.custom_outputs?.trace_id),d=document.createElement('article');d.className='turn assistant';
 d.innerHTML='<div class="answer-heading"><span class="brandmark">P</span>Databricks specialist workflow</div><div class="answer-body">'+markdown(answer)+genieEvidence(data.custom_outputs?.genie_result)+'</div><div class="answer-meta"><span>AI-generated · Verify before use</span><span>'+seconds+'s</span>'+(url?'<a href="'+escapeHTML(url)+'" target="_blank" rel="noopener noreferrer">View execution ↗</a>':'')+'</div>';
 el('chat').append(d);window.GenieCharts?.mount(d,data.custom_outputs?.genie_result?.tables||[]);el('trace').hidden=!url;if(url)el('trace').href=url;
 return answer;
}
function setBusy(busy){state.busy=busy;['send','new-chat'].forEach(id=>el(id).disabled=busy);document.querySelectorAll('[data-prompt]').forEach(b=>b.disabled=busy);el('stop').hidden=!busy;el('send').innerHTML=busy?'Working…':'Send <span>↑</span>';el('chat').setAttribute('aria-busy',String(busy))}
el('stop').onclick=()=>state.controller?.abort();
el('question').addEventListener('keydown',e=>{
 if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!state.busy)el('form').requestSubmit()}
});
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{
 if(state.busy)return;el('question').value=b.dataset.prompt;el('question').focus();
});
el('form').onsubmit=async e=>{
 e.preventDefault();const question=el('question').value.trim();if(!question||state.busy)return;
 const messages=[...state.history.slice(-28),{role:'user',content:question}];
 el('question').value='';el('plan-card').hidden=true;el('welcome').hidden=true;el('error').hidden=true;el('trace').hidden=true;showUser(question);state.controller=new AbortController();setBusy(true);state.started=Date.now();activity('accepted');
 const pending=document.createElement('div');pending.className='pending-response';pending.innerHTML='<span class="spinner" aria-hidden="true"></span><span>Connecting to your agent team…</span>';el('chat').append(pending);
 let complete=false;state.timer=setInterval(()=>{const seconds=Math.floor((Date.now()-state.started)/1000);el('duration').textContent='Elapsed time: '+seconds+'s'},1000);
 try{
  await window.Conversations.user(question);
  const response=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,conversation_id:window.Conversations.conversationId}),signal:state.controller.signal});
  if(!response.ok){let detail;try{detail=(await response.json()).detail}catch{}throw Error(typeof detail==='string'?detail:'The request could not be accepted. Check your connection and try again.')}
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
  async function onEvent(block){let kind='message',parts=[];for(const line of block.split('\n')){if(line.startsWith('event:'))kind=line.slice(6).trim();if(line.startsWith('data:'))parts.push(line.slice(5).trimStart())}if(!parts.length)return;const data=JSON.parse(parts.join('\n')),seconds=data.elapsed_seconds??Math.floor((Date.now()-state.started)/1000);
   if(kind==='accepted')activity('accepted',seconds);
   if(kind==='activity'){executionActivity(data,seconds);pending.lastElementChild.textContent=data.message||'Executing the selected step'}
   if(kind==='error'){const url=traceURL(data.trace_id);el('trace').hidden=!url;if(url)el('trace').href=url;const link=genieLink(data);if(link){const d=document.createElement('p'),a=document.createElement('a');a.href=link;a.target='_blank';a.rel='noopener noreferrer';a.textContent='Check the Genie conversation ↗';d.append(a);el('chat').append(d)}throw Error(data.message);}
   if(kind==='result'){pending.remove();const answer=showAnswer(data,seconds);state.history=[...messages,{role:'assistant',content:answer}];complete=true;activity('done',seconds);await window.Conversations.answer(answer,data,seconds)}
  }
  while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});let end;while((end=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,end);buffer=buffer.slice(end+2);await onEvent(block)}if(done)break}
  if(!complete)throw Error('The connection ended before a response arrived.'+' Inspect the trace before submitting another request.');
 }catch(err){pending.remove();const stopped=err.name==='AbortError';el('error').textContent=stopped?'Stopped waiting for this response. An in-flight model or tool call may still finish.':err.message;el('error').hidden=false;if(complete){el('error').textContent='Answer received, but saving history failed: '+err.message}else{try{await window.Conversations.interrupted(el('error').textContent)}catch{}activity(stopped?'stopped':'failed',Math.floor((Date.now()-state.started)/1000));}}
 finally{clearInterval(state.timer);state.controller=null;setBusy(false);el('question').focus()}
};
fetch('/api/whoami').then(r=>{if(!r.ok)throw Error();return r.json()}).then(d=>{el('identity').textContent=d.user||'Authenticated demo user';state.experiment=d.experiment_id;state.workspaceId=d.workspace_id;try{const u=new URL(d.workspace_url);if(u.protocol==='https:')state.workspace=u.origin}catch{}if(window.Conversations)window.Conversations.init();else document.addEventListener('DOMContentLoaded',()=>window.Conversations.init(),{once:true})}).catch(()=>{el('identity').textContent='Account details unavailable'});
