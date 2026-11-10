'use strict';
window.Conversations=(()=>{
 let current=null,revision=0,turns=[],ready=false,dirty=false,title=null,renaming=null;
 let preferences={display_name:'',chart_type:'bar'};
 const status=text=>{el('history-status').textContent=text};
 async function api(url,options={}){
  const r=await fetch(url,{...options,headers:{'Content-Type':'application/json'}});
  if(!r.ok){let detail;try{detail=(await r.json()).detail}catch{}throw Error(typeof detail==='string'?detail:'Saved history is unavailable. Try again shortly.')}
  return r.json();
 }
 async function refresh(){
  const data=await api('/api/conversations');el('conversation-list').replaceChildren();
  data.conversations.forEach(item=>{
   const row=document.createElement('div');row.className='conversation-row';
   const b=document.createElement('button');b.className='saved-conversation'+(item.id===current?' selected':'');b.textContent=item.title;b.title=item.title+' · '+item.updated_at;b.onclick=()=>open(item.id);
   const edit=document.createElement('button');edit.className='rename-conversation';edit.textContent='✎';edit.title='Rename conversation';edit.setAttribute('aria-label','Rename '+item.title);edit.onclick=()=>{
    if(state.busy){status('Wait for the current response before renaming.');return}
    renaming=item;el('conversation-name').value=item.title;el('rename-error').textContent='';el('rename-dialog').showModal();el('conversation-name').select();
   };
   row.append(b,edit);el('conversation-list').append(row);
  });
 }
 async function save(){
  if(!ready)throw Error('Saved history is unavailable. Retry loading history before sending.');
  if(!turns.length)return;
  if(!current)current=crypto.randomUUID();
  dirty=true;status('Saving…');
  try{
   if(!title)title=String(turns.find(t=>t.role==='user')?.content||'Conversation').slice(0,120);
   const data=await api('/api/conversations/'+current,{method:'PUT',body:JSON.stringify({title,revision,turns})});revision=data.revision;dirty=false;status('All changes saved');
  }catch(e){status('Not saved: '+e.message);throw e}
  try{await refresh()}catch{status('Conversation saved; list refresh failed')}
 }
 async function open(id){
  if(state.busy)return;
  try{
   if(dirty)await save();
   const data=await api('/api/conversations/'+id);
   reset();current=data.id;revision=data.revision;turns=data.turns;title=data.title;dirty=false;
   el('welcome').hidden=true;
   for(const turn of turns){
    if(turn.role==='user')showUser(turn.content);
    else if(turn.role==='assistant')showAnswer(turn.response,turn.seconds||0);
    else notice(turn.content);
   }
   state.history=turns.filter(t=>t.role==='user'||t.role==='assistant').map(t=>({role:t.role,content:t.content}));
   status('Conversation loaded');document.body.classList.remove('history-open');el('history-toggle').setAttribute('aria-expanded','false');await refresh();
  }catch(e){status(e.message)}
 }
 function notice(text){const p=document.createElement('p');p.className='saved-notice';p.textContent=text;el('chat').append(p)}
 function reset(){
  state.history=[];window.GenieCharts?.cleanup(el('chat'));el('chat').replaceChildren();el('welcome').hidden=false;el('error').hidden=true;el('trace').hidden=true;el('duration').hidden=true;el('plan-card').hidden=true;el('activity-badge').textContent='Ready';el('activity-badge').className='badge';el('timeline').replaceChildren();el('question').value='';state.events=[];
 }
 async function start(){if(state.busy)return;try{if(dirty)await save();reset();current=null;revision=0;turns=[];title=null;document.body.classList.remove('history-open');el('history-toggle').setAttribute('aria-expanded','false');status('New conversation · previous threads are saved');await refresh();el('question').focus()}catch(e){status(e.message)}}
 async function init(){
  try{preferences=await api('/api/preferences');ready=true;await refresh();status('Your saved conversations');el('profile-name').value=preferences.display_name;el('profile-chart').value=preferences.chart_type;if(preferences.display_name)el('identity').textContent=preferences.display_name}
  catch(e){status(e.message)}
 }
 el('history-toggle').onclick=()=>{const opened=document.body.classList.toggle('history-open');el('history-toggle').setAttribute('aria-expanded',String(opened))};
 el('new-chat').onclick=start;
 el('rename-cancel').onclick=()=>el('rename-dialog').close();
 el('rename-form').onsubmit=async e=>{
  e.preventDefault();if(!renaming||state.busy)return;
  const name=el('conversation-name').value.trim();if(!name){el('rename-error').textContent='Enter a conversation name';return}
  el('rename-save').disabled=true;
  try{
   // Save pending turns before applying the title-only update.
   if(dirty)await save();
   const selected=renaming;
   const data=await api('/api/conversations/'+selected.id,{method:'PATCH',body:JSON.stringify({title:name,revision:selected.id===current?revision:selected.revision})});
   if(selected.id===current){title=data.title;revision=data.revision}
   el('rename-dialog').close();status('Conversation renamed');await refresh();
  }catch(e){el('rename-error').textContent=e.message}
  finally{el('rename-save').disabled=false}
 };
 el('history-retry').onclick=init;
 el('preferences-open').onclick=()=>el('preferences-dialog').showModal();
 el('preferences-close').onclick=()=>el('preferences-dialog').close();
 el('preferences-form').onsubmit=async e=>{e.preventDefault();try{preferences=await api('/api/preferences',{method:'PUT',body:JSON.stringify({display_name:el('profile-name').value.trim(),chart_type:el('profile-chart').value})});el('preferences-dialog').close();status('Preferences saved')}catch(e){el('preferences-error').textContent=e.message}};
 return {init,start,get chartType(){return preferences.chart_type},
  async user(content){turns.push({role:'user',content});await save()},
  async answer(content,response,seconds){turns.push({role:'assistant',content,response,seconds});await save()},
  async interrupted(message){turns.push({role:'notice',content:message});await save()}
 };
})();
