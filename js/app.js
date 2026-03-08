import { EXAMPLES } from './examples.js';

// =====================================================================
//  TOKENIZER
// =====================================================================
const TT={NUM:'NUM',ID:'ID',PLUS:'+',MINUS:'-',STAR:'*',SLASH:'/',LP:'(',RP:')',COMMA:',',EOF:'EOF'};
function tokenize(e){const t=[];let i=0;while(i<e.length){if(/\s/.test(e[i])){i++;continue}
if(/\d/.test(e[i])||(e[i]==='.'&&i+1<e.length&&/\d/.test(e[i+1]))){let n='';while(i<e.length&&(/\d/.test(e[i])||e[i]==='.'))n+=e[i++];t.push({type:TT.NUM,value:n});continue}
if(/[a-zA-Z_]/.test(e[i])){let n='';while(i<e.length&&/[a-zA-Z0-9_]/.test(e[i]))n+=e[i++];t.push({type:TT.ID,value:n});continue}
const c=e[i];if('+-*/(),'.includes(c)){t.push({type:c==='+'?TT.PLUS:c==='-'?TT.MINUS:c==='*'?TT.STAR:c==='/'?TT.SLASH:c==='('?TT.LP:c===')'?TT.RP:TT.COMMA,value:c});i++;continue}
throw new Error(`Bad char '${c}' at ${i}`);}t.push({type:TT.EOF,value:''});return t;}

// =====================================================================
//  PARSER
// =====================================================================
function parse(t){let p=0;const pk=()=>t[p],ad=()=>t[p++],ex=c=>{const k=ad();if(k.type!==c)throw new Error(`Expected '${c}' got '${k.type}'`);return k};
function expr(){let l=term();while(pk().type===TT.PLUS||pk().type===TT.MINUS){const o=ad().type;l={type:'bin',op:o===TT.PLUS?'add':'sub',l,r:term()}}return l}
function term(){let l=un();while(pk().type===TT.STAR||pk().type===TT.SLASH){const o=ad().type;l={type:'bin',op:o===TT.STAR?'mul':'div',l,r:un()}}return l}
function un(){if(pk().type===TT.MINUS){ad();return{type:'un',op:'neg',a:un()}}return call()}
function call(){if(pk().type===TT.ID&&p+1<t.length&&t[p+1].type===TT.LP){const n=ad().value;ad();const a=[];
if(pk().type!==TT.RP){a.push(expr());while(pk().type===TT.COMMA){ad();a.push(expr())}}ex(TT.RP);return{type:'fn',name:n,a}}return pri()}
function pri(){const k=pk();if(k.type===TT.NUM){ad();return{type:'num',v:parseFloat(k.value)}}
if(k.type===TT.ID){ad();return{type:'var',name:k.value}}if(k.type===TT.LP){ad();const e=expr();ex(TT.RP);return e}
throw new Error(`Unexpected '${k.value||k.type}'`);}
const r=expr();if(pk().type!==TT.EOF)throw new Error(`Extra tokens: '${pk().value}'`);return r;}

// =====================================================================
//  GRAPH
// =====================================================================
class GN{constructor(id,tp,op,label,ins=[]){this.id=id;this.tp=tp;this.op=op;this.label=label;this.ins=ins;
this.val=null;this.grad=0;this.rank=-1;this.x=0;this.y=0;this.fD=false;this.bD=false;this.fA=false;this.bA=false}}
const OPS={add:'+',sub:'\u2212',mul:'\u00D7',div:'\u00F7'};
const FNS=new Set(['exp','log','sigmoid','relu','tanh','sin','cos','sqrt','abs','neg']);
function buildG(ast){const ns=[],vs={},vo=[];let id=0;
function mk(tp,op,lb,ins=[]){const n=new GN(id++,tp,op,lb,ins);ns.push(n);return n}
function w(a){
if(a.type==='num'){const n=mk('c',null,fc(a.v));n.val=a.v;return n}
if(a.type==='var'){if(!vs[a.name]){vs[a.name]=mk('i',null,a.name);vo.push(a.name)}return vs[a.name]}
if(a.type==='bin'){return mk('o',a.op,OPS[a.op]||a.op,[w(a.l),w(a.r)])}
if(a.type==='un'){return mk('o','neg','\u2212()',[w(a.a)])}
if(a.type==='fn'){if(!FNS.has(a.name))throw new Error(`Unknown function: ${a.name}`);return mk('o',a.name,a.name,a.a.map(w))}}
return{ns,vs,vo,out:w(ast)}}
function fc(v){if(Number.isInteger(v)&&Math.abs(v)<1e4)return String(v);return parseFloat(v.toPrecision(5)).toString()}
function fm(v){if(v==null)return'?';if(!isFinite(v))return String(v);if(v===0)return'0';if(Math.abs(v)<1e-6)return v.toExponential(2);return parseFloat(v.toPrecision(5)).toString()}

// =====================================================================
//  FORWARD & BACKWARD
// =====================================================================
function compV(n){const a=n.ins[0]?.val,b=n.ins[1]?.val;switch(n.op){
case'add':return a+b;case'sub':return a-b;case'mul':return a*b;case'div':return a/b;case'neg':return-a;
case'exp':return Math.exp(a);case'log':return Math.log(a);case'sigmoid':return 1/(1+Math.exp(-a));
case'relu':return Math.max(0,a);case'tanh':return Math.tanh(a);case'sin':return Math.sin(a);
case'cos':return Math.cos(a);case'sqrt':return Math.sqrt(a);case'abs':return Math.abs(a)}}

function lgrads(n){const u=n.grad,a=n.ins[0]?.val,b=n.ins[1]?.val;switch(n.op){
case'add':return[{t:n.ins[0],g:u,l:1},{t:n.ins[1],g:u,l:1}];
case'sub':return[{t:n.ins[0],g:u,l:1},{t:n.ins[1],g:-u,l:-1}];
case'mul':return[{t:n.ins[0],g:u*b,l:b},{t:n.ins[1],g:u*a,l:a}];
case'div':return[{t:n.ins[0],g:u/b,l:1/b},{t:n.ins[1],g:-u*a/(b*b),l:-a/(b*b)}];
case'neg':return[{t:n.ins[0],g:-u,l:-1}];
case'exp':return[{t:n.ins[0],g:u*n.val,l:n.val}];
case'log':return[{t:n.ins[0],g:u/a,l:1/a}];
case'sigmoid':return[{t:n.ins[0],g:u*n.val*(1-n.val),l:n.val*(1-n.val)}];
case'relu':return[{t:n.ins[0],g:u*(a>0?1:0),l:a>0?1:0}];
case'tanh':return[{t:n.ins[0],g:u*(1-n.val*n.val),l:1-n.val*n.val}];
case'sin':return[{t:n.ins[0],g:u*Math.cos(a),l:Math.cos(a)}];
case'cos':return[{t:n.ins[0],g:u*(-Math.sin(a)),l:-Math.sin(a)}];
case'sqrt':return[{t:n.ins[0],g:u/(2*n.val),l:1/(2*n.val)}];
case'abs':return[{t:n.ins[0],g:u*(a>=0?1:-1),l:a>=0?1:-1}];default:return[]}}

function gfDesc(n){switch(n.op){
case'add':return['d(a+b)/da = 1','d(a+b)/db = 1'];
case'sub':return['d(a-b)/da = 1','d(a-b)/db = -1'];
case'mul':return[`d(a\u00D7b)/da = b = ${fm(n.ins[1]?.val)}`,`d(a\u00D7b)/db = a = ${fm(n.ins[0]?.val)}`];
case'div':return[`d(a/b)/da = 1/b`,`d(a/b)/db = -a/b\u00B2`];
case'neg':return['d(-a)/da = -1'];case'exp':return[`d(e^a)/da = e^a = ${fm(n.val)}`];
case'log':return[`d(ln a)/da = 1/a = ${fm(1/n.ins[0]?.val)}`];
case'sigmoid':return[`\u03C3\'= \u03C3(1-\u03C3) = ${fm(n.val*(1-n.val))}`];
case'relu':return[`relu\' = ${n.ins[0]?.val>0?'1':'0'}`];
case'tanh':return[`tanh\' = 1-tanh\u00B2 = ${fm(1-n.val*n.val)}`];default:return['?']}}

function topo(out){const v=new Set(),o=[];function go(n){if(v.has(n.id))return;v.add(n.id);n.ins.forEach(go);o.push(n)}go(out);return o}

// =====================================================================
//  LAYOUT
// =====================================================================
function avg(a){return a.length?a.reduce((s,v)=>s+v,0)/a.length:0}
function nodeGeom(n){
  if(n.tp==='o')return{shape:'circle',r:34,w:68,h:68};
  return{shape:'rect',w:Math.max(108,n.label.length*11+38),h:56,rx:16}
}
function layoutG(g){
  const{ns}=g,mem={};
  function rk(n){if(mem[n.id]!=null)return mem[n.id];if(!n.ins.length)return mem[n.id]=0;return mem[n.id]=Math.max(...n.ins.map(rk))+1}
  ns.forEach(n=>{n.rank=rk(n);n.geom=nodeGeom(n)});
  const mxR=Math.max(...ns.map(n=>n.rank));
  const ranks=Array.from({length:mxR+1},()=>[]);
  ns.forEach(n=>ranks[n.rank].push(n));

  const M=104,GX=126,GY=76;
  const rankH=r=>r.reduce((s,n)=>s+n.geom.h,0)+Math.max(0,r.length-1)*GY;
  const rankW=r=>Math.max(136,...r.map(n=>n.geom.w));
  const totalH=Math.max(260,...ranks.map(rankH));
  const widths=ranks.map(rankW);
  const xs=[];let xCur=M;
  widths.forEach((w,i)=>{xs[i]=xCur+w/2;xCur+=w+(i<widths.length-1?GX:0)});

  function placeRank(rank,idx){
    let y=M+(totalH-rankH(rank))/2;
    rank.forEach(n=>{y+=n.geom.h/2;n.x=xs[idx];n.y=y;y+=n.geom.h/2+GY})
  }

  ranks.forEach((rank,idx)=>placeRank(rank,idx));
  for(let pass=0;pass<2;pass++){
    for(let r=1;r<=mxR;r++){
      ranks[r].sort((a,b)=>avg(a.ins.map(n=>n.y))-avg(b.ins.map(n=>n.y)));
      placeRank(ranks[r],r)
    }
    const con={};
    ns.forEach(n=>{if(n.rank>0)n.ins.forEach(inp=>{if(inp.rank===0)(con[inp.id]=con[inp.id]||[]).push(n)})});
    ranks[0].sort((a,b)=>avg((con[a.id]||[]).map(n=>n.y))-avg((con[b.id]||[]).map(n=>n.y)));
    placeRank(ranks[0],0)
  }
  const totalW=M*2+widths.reduce((s,w)=>s+w,0)+GX*mxR;
  return{w:totalW,h:totalH+M*2}
}

// =====================================================================
//  SVG RENDERER
// =====================================================================
const NS='http://www.w3.org/2000/svg';
function sv(tag,a={},txt){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(txt!=null)e.textContent=txt;return e}

function ePt(nd,tx,ty){const dx=tx-nd.x,dy=ty-nd.y,d=Math.hypot(dx,dy);if(d===0)return{x:nd.x,y:nd.y};
const gm=nd.geom||nodeGeom(nd);
if(gm.shape==='circle')return{x:nd.x+dx/d*gm.r,y:nd.y+dy/d*gm.r}
const hw=gm.w/2,hh=gm.h/2,sx=dx?hw/Math.abs(dx):1e9,sy=dy?hh/Math.abs(dy):1e9,s=Math.min(sx,sy);
return{x:nd.x+dx*s,y:nd.y+dy*s}}
function qPath(c){return`M ${c.from.x} ${c.from.y} Q ${c.ctrl.x} ${c.ctrl.y} ${c.to.x} ${c.to.y}`}
function qPoint(c,t){const u=1-t;return{x:u*u*c.from.x+2*u*t*c.ctrl.x+t*t*c.to.x,y:u*u*c.from.y+2*u*t*c.ctrl.y+t*t*c.to.y}}
function qTan(c,t){return{x:2*(1-t)*(c.ctrl.x-c.from.x)+2*t*(c.to.x-c.ctrl.x),y:2*(1-t)*(c.ctrl.y-c.from.y)+2*t*(c.to.y-c.ctrl.y)}}
function qOffset(c,t,d){const p=qPoint(c,t),tg=qTan(c,t),len=Math.hypot(tg.x,tg.y)||1;return{x:p.x+(-tg.y/len)*d,y:p.y+(tg.x/len)*d}}
function graphBounds(g){
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  g.ns.forEach(n=>{
    const gm=n.geom||nodeGeom(n),extra=(n.fD||n.bD)?34:18;
    minX=Math.min(minX,n.x-gm.w/2-extra);maxX=Math.max(maxX,n.x+gm.w/2+extra);
    minY=Math.min(minY,n.y-gm.h/2-extra);maxY=Math.max(maxY,n.y+gm.h/2+extra)
  });
  return{x:minX,y:minY,w:maxX-minX,h:maxY-minY}
}
function backCurve(src,dst,idx,count){
  const from=ePt(src,dst.x,dst.y),to=ePt(dst,src.x,src.y),dx=to.x-from.x,dy=to.y-from.y,len=Math.hypot(dx,dy)||1;
  const nx=-dy/len,ny=dx/len,spread=count===1?0:idx-(count-1)/2;
  const sign=spread===0?(dy>=0?1:-1):Math.sign(spread);
  const lift=(Math.max(26,Math.min(72,len*0.18))+Math.abs(spread)*16)*sign;
  return{from,to,ctrl:{x:(from.x+to.x)/2+nx*lift,y:(from.y+to.y)/2+ny*lift},spread,sign}
}
function clamp(v,min,max){return Math.max(min,Math.min(max,v))}
function baseViewFor(g,bwdAnn){
  const b=graphBounds(g),pad=bwdAnn?132:92;
  return{x:b.x-pad,y:b.y-pad,w:b.w+pad*2,h:b.h+pad*2}
}
function activeGraphNode(){return!S.g?null:S.g.ns.find(n=>n.bA||n.fA)||S.g.out}
function clampCamera(){
  if(!S.cam.base||!S.cam.current)return;
  const base=S.cam.base,cur=S.cam.current,zoom=clamp(base.w/cur.w,.85,5);
  cur.w=base.w/zoom;cur.h=base.h/zoom;
  const padX=Math.max(72,cur.w*.18),padY=Math.max(72,cur.h*.18);
  const minX=base.x-padX,maxX=base.x+base.w-cur.w+padX;
  const minY=base.y-padY,maxY=base.y+base.h-cur.h+padY;
  cur.x=clamp(cur.x,Math.min(minX,maxX),Math.max(minX,maxX));
  cur.y=clamp(cur.y,Math.min(minY,maxY),Math.max(minY,maxY))
}
function syncCamera(base){
  if(!S.cam.base||!S.cam.current){S.cam.base=base;S.cam.current={...base};return}
  const oldBase=S.cam.base,old=S.cam.current;S.cam.base=base;
  if(!S.cam.manual){S.cam.current={...base};return}
  const zoom=clamp(oldBase.w/old.w,.85,5),cx=old.x+old.w/2,cy=old.y+old.h/2;
  S.cam.current={x:cx-base.w/(2*zoom),y:cy-base.h/(2*zoom),w:base.w/zoom,h:base.h/zoom};
  clampCamera()
}
function zoomPct(){return!S.cam.base||!S.cam.current?100:Math.round((S.cam.base.w/S.cam.current.w)*100)}
function updateViewportHud(){
  const on=!!S.g;
  $('graph-toolbar').style.display=on?'flex':'none';
  $('graph-note').style.display=on?'block':'none';
  $('zoom-pill').textContent=`${zoomPct()}%`;
  $('btn-focus').disabled=!activeGraphNode()
}
function requestRender(reason='',emit=false){if(!S.g)return;renderG($('graph-svg'),S.g,S.dm,S.bwdAnn);if(emit)emitStateChange(reason)}
function fitGraph(emit=true){
  if(!S.g||!S.cam.base)return;
  S.cam.manual=false;S.cam.current={...S.cam.base};requestRender('fit-view',emit)
}
function zoomGraph(mult,point=null,emit=false){
  if(!S.g||!S.cam.base||!S.cam.current)return;
  const cur=S.cam.current,base=S.cam.base,zoom=clamp(base.w/cur.w*mult,.85,5),nextW=base.w/zoom,nextH=base.h/zoom;
  const ctr=point||{x:cur.x+cur.w/2,y:cur.y+cur.h/2},rx=(ctr.x-cur.x)/cur.w,ry=(ctr.y-cur.y)/cur.h;
  S.cam.current={x:ctr.x-rx*nextW,y:ctr.y-ry*nextH,w:nextW,h:nextH};S.cam.manual=true;clampCamera();requestRender('zoom-view',emit)
}
function focusNode(node=activeGraphNode(),emit=true){
  if(!node||!S.cam.base)return;
  const zoom=clamp(S.cam.base.w/Math.max(260,(node.geom?.w||140)*3),1.35,2.4),w=S.cam.base.w/zoom,h=S.cam.base.h/zoom;
  S.cam.current={x:node.x-w/2,y:node.y-h/2,w,h};S.cam.manual=true;clampCamera();requestRender('focus-node',emit)
}
function clientToGraph(svg,x,y){
  const r=svg.getBoundingClientRect(),vb=S.cam.current||S.cam.base;
  return{x:vb.x+((x-r.left)/r.width)*vb.w,y:vb.y+((y-r.top)/r.height)*vb.h}
}

function renderG(svg,g,dims,bwdAnn){
  svg.innerHTML='';
  syncCamera(baseViewFor(g,bwdAnn));
  const vb=S.cam.current||S.cam.base;
  svg.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
  svg.setAttribute('preserveAspectRatio','xMidYMid meet');

  const defs=sv('defs');
  [['ah','#94a3b8'],['ahg','#10b981'],['ahr','#f43f5e'],['ahv','#7c3aed']].forEach(([id,c])=>{
    const m=sv('marker',{id,markerWidth:10,markerHeight:7,refX:9,refY:3.5,orient:'auto',markerUnits:'strokeWidth'});
    m.appendChild(sv('polygon',{points:'0 0.5,10 3.5,0 6.5',fill:c}));defs.appendChild(m)});
  const ft=sv('filter',{id:'gl',x:'-50%',y:'-50%',width:'200%',height:'200%'});
  ft.appendChild(sv('feGaussianBlur',{stdDeviation:'4',result:'b'}));
  const mg=sv('feMerge');mg.appendChild(sv('feMergeNode',{in:'b'}));mg.appendChild(sv('feMergeNode',{in:'SourceGraphic'}));
  ft.appendChild(mg);defs.appendChild(ft);svg.appendChild(defs);

  // Edges
  const eG=sv('g');
  g.ns.forEach(n=>n.ins.forEach(inp=>{
    const f=ePt(inp,n.x,n.y),t=ePt(n,inp.x,inp.y);
    const isActive=n.bA;const done=inp.fD&&n.fD;
    const col=isActive?'#fda4af':done?'#86efac':'#e2e8f0';
    const w=done?2.2:1.6;const ah=done?'url(#ahg)':'url(#ah)';
    eG.appendChild(sv('line',{x1:f.x,y1:f.y,x2:t.x,y2:t.y,stroke:col,'stroke-width':w,'marker-end':ah,'stroke-linecap':'round',opacity:done?.92:.82}))}));
  svg.appendChild(eG);

  // Nodes
  const nG=sv('g');
  g.ns.forEach(n=>{
    const grp=sv('g',{transform:`translate(${n.x},${n.y})`,'data-nid':n.id,class:'gnode',style:'cursor:pointer'});
    const gm=n.geom||nodeGeom(n);
    let fill,stroke,sw=2;
    if(n.fA){fill='#a7f3d0';stroke='#059669';sw=3;grp.setAttribute('filter','url(#gl)')}
    else if(n.bA){fill='#fecdd3';stroke='#e11d48';sw=3;grp.setAttribute('filter','url(#gl)')}
    else if(n.bD&&n.tp!=='c'){fill='#ffe4e6';stroke='#fb7185'}
    else if(n.fD){fill='#d1fae5';stroke='#34d399'}
    else if(n.tp==='i'){fill='#e0e7ff';stroke='#818cf8'}
    else if(n.tp==='c'){fill='#fef3c7';stroke='#fbbf24'}
    else{fill='#f1f5f9';stroke='#94a3b8'}

    if(gm.shape==='circle')grp.appendChild(sv('circle',{r:gm.r,fill,stroke,'stroke-width':sw}));
    else grp.appendChild(sv('rect',{x:-gm.w/2,y:-gm.h/2,width:gm.w,height:gm.h,rx:gm.rx,fill,stroke,'stroke-width':sw}));
    const fs=n.tp==='o'?(n.label.length>4?11:n.label.length>3?13:16):14;
    grp.appendChild(sv('text',{'text-anchor':'middle','dominant-baseline':'central','font-size':fs,'font-weight':'700',fill:'#334155'},n.label));

    if(n.fD&&n.val!=null)grp.appendChild(badge(0,gm.h/2+18,fm(n.val),'#dcfce7','#047857',10,'fade-up'));
    if(n.bD&&n.tp!=='c')grp.appendChild(badge(0,-(gm.h/2+18),`\u2207 ${fm(n.grad)}`,'#ffe4e6','#be123c',10,'fade-up'));
    nG.appendChild(grp)});
  svg.appendChild(nG);

  // ======================== BACKWARD ANNOTATIONS ========================
  if(bwdAnn){
    const ann=sv('g',{class:'bwd-ann'});
    const nd=bwdAnn.node;
    const gm=nd.geom||nodeGeom(nd);

    // Upstream badge above active node
    ann.appendChild(badge(nd.x,nd.y-(gm.h/2+30),`up ${fm(bwdAnn.upstream)}`,'#e11d48','#fff',10,'fade-up'));

    // For each input: backward arrow + gradient badge
    bwdAnn.items.forEach((item,idx)=>{
      const curve=backCurve(nd,item.target,idx,bwdAnn.items.length);
      const offsetDir=curve.sign*(18+Math.abs(curve.spread)*4);
      ann.appendChild(sv('path',{d:qPath(curve),fill:'none',stroke:'#7c3aed','stroke-width':2.5,'stroke-dasharray':'7 4',
        'marker-end':'url(#ahv)',opacity:.88,class:'pulse-glow','stroke-linecap':'round'}));
      const lp=qOffset(curve,.36,offsetDir),dp=qOffset(curve,.72,offsetDir*1.15);
      ann.appendChild(badge(lp.x,lp.y,`\u2202 ${fm(item.local)}`,'#f59e0b','#78350f',9,'fade-up'));
      ann.appendChild(badge(dp.x,dp.y,`\u2207 ${fm(item.downstream)}`,'#7c3aed','#fff',9,'fade-up'));
    });

    svg.appendChild(ann);
  }

  // Tooltip events
  nG.querySelectorAll('.gnode').forEach(grp=>{
    grp.addEventListener('mouseenter',e=>{const nid=parseInt(grp.dataset.nid);showTT(e,g.ns.find(n=>n.id===nid))});
    grp.addEventListener('mousemove',e=>{const tt=$('tooltip');tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY+12)+'px'});
    grp.addEventListener('mouseleave',()=>{$('tooltip').style.display='none'})});
  updateViewportHud()
}

function badge(cx,cy,text,bg,fg,fs=10,cls=''){
  const g=sv('g',cls?{class:cls}:{});
  const tw=text.length*fs*0.56+16,th=fs+10;
  g.appendChild(sv('rect',{x:cx-tw/2,y:cy-th/2,width:tw,height:th,rx:th/2,fill:bg,opacity:.93,
    filter:'drop-shadow(0 1px 3px rgba(0,0,0,0.2))'}));
  g.appendChild(sv('text',{x:cx,y:cy+1,'text-anchor':'middle','dominant-baseline':'central',
    'font-size':fs,'font-weight':'700',fill:fg,'font-family':'IBM Plex Mono,monospace'},text));
  return g}

function showTT(e,n){if(!n)return;const tt=$('tooltip');
let h=`<b style="color:#a5b4fc">${n.tp==='o'?n.op:n.label}</b> <span style="opacity:.4">#${n.id}</span>`;
if(n.val!=null)h+=`<br>Val: <span style="color:#6ee7b7">${fm(n.val)}</span>`;
if(n.bD&&n.tp!=='c')h+=`<br>Grad: <span style="color:#fca5a5">${fm(n.grad)}</span>`;
tt.innerHTML=h;tt.style.display='block';tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY+12)+'px'}

// =====================================================================
//  EXAMPLES
// =====================================================================
const EX = EXAMPLES;

// =====================================================================
//  APP
// =====================================================================
let S={g:null,tp:[],fi:0,bi:0,ph:'idle',dm:null,bwdAnn:null,cam:{base:null,current:null,manual:false,drag:false,last:null,moved:false}};
let pTimer=null;
const $=id=>document.getElementById(id);
const subs=new Set();

function snapshotPayload({g,tp,fi,bi,ph,dm,bwdAnn},expr,vars,infoText='',reason='snapshot'){
  return{
    savedAt:new Date().toISOString(),
    reason,
    expression:expr,
    phase:ph,
    forwardIndex:fi,
    backwardIndex:bi,
    totalSteps:tp.length,
    outputId:g.out.id,
    outputValue:g.out.val,
    variables:{...vars},
    infoText,
    dimensions:dm?{width:dm.w,height:dm.h}:null,
    viewport:S.cam.base&&S.cam.current?{
      base:{...S.cam.base},
      current:{...S.cam.current},
      zoom:zoomPct()
    }:null,
    backwardAnnotation:bwdAnn?{
      nodeId:bwdAnn.node.id,
      upstream:bwdAnn.upstream,
      items:bwdAnn.items.map(it=>({targetId:it.target.id,local:it.local,downstream:it.downstream,desc:it.desc}))
    }:null,
    nodes:g.ns.map(n=>({
      id:n.id,type:n.tp,op:n.op,label:n.label,inputIds:n.ins.map(inp=>inp.id),
      rank:n.rank,position:{x:n.x,y:n.y},value:n.val,grad:n.grad,
      forwardDone:n.fD,backwardDone:n.bD,forwardActive:n.fA,backwardActive:n.bA
    }))
  }
}
function varsNow(){const out={};$('var-bar').querySelectorAll('input').forEach(inp=>out[inp.dataset.v]=parseFloat(inp.value)||0);return out}
function annState(){if(!S.bwdAnn)return null;return{
  nodeId:S.bwdAnn.node.id,
  upstream:S.bwdAnn.upstream,
  items:S.bwdAnn.items.map(it=>({targetId:it.target.id,local:it.local,downstream:it.downstream,desc:it.desc}))
}}
function graphSnapshot(){
  if(!S.g)return null;
  return snapshotPayload({g:S.g,tp:S.tp,fi:S.fi,bi:S.bi,ph:S.ph,dm:S.dm,bwdAnn:S.bwdAnn},$('expr-input').value.trim(),varsNow(),$('info-box').textContent)
}
function serializeCurrentSvg(){
  if(!S.g)return null;
  const svg=$('graph-svg').cloneNode(true);
  svg.setAttribute('xmlns',NS);
  return new XMLSerializer().serializeToString(svg)
}
function downloadText(text,name,type){
  const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),0);
  return name
}
function fileStamp(){return new Date().toISOString().replace(/[:.]/g,'-')}
function saveGraphJSON(name=`autograd-graph-${fileStamp()}.json`){
  const snap=graphSnapshot();if(!snap)return null;
  downloadText(JSON.stringify(snap,null,2),name,'application/json');return snap
}
function saveGraphSvg(name=`autograd-graph-${fileStamp()}.svg`){
  const text=serializeCurrentSvg();if(!text)return null;
  downloadText(text,name,'image/svg+xml');return text
}
function stageTimeline(){
  if(!S.g)return null;
  const expr=$('expr-input').value.trim(),vars=varsNow(),ast=parse(tokenize(expr)),g=buildG(ast),tp=topo(g.out),dm=layoutG(g);
  Object.entries(vars).forEach(([nm,v])=>{if(g.vs[nm])g.vs[nm].val=v});
  let fi=0,bi=0,ph='idle',bwdAnn=null;
  const stages=[];
  const snap=(reason,infoText='')=>stages.push(snapshotPayload({g,tp,fi,bi,ph,dm,bwdAnn},expr,vars,infoText,reason));
  g.ns.forEach(n=>{n.fD=false;n.bD=false;n.fA=false;n.bA=false;n.grad=0;if(n.tp!=='c'&&n.tp!=='i')n.val=null});
  snap('idle','Initial graph state');
  while(fi<tp.length){
    const nd=tp[fi];
    if(nd.tp!=='i'&&nd.tp!=='c')nd.val=compV(nd);
    nd.fD=true;nd.fA=true;ph=fi+1>=tp.length?'fwd_done':'forward';
    snap('forward-step',nd.tp==='o'?`Forward ${fi+1}: ${nd.op}`:`Forward ${fi+1}: ${nd.label}`);nd.fA=false;fi++
  }
  ph='backward';g.out.grad=1;
  const rev=[...tp].reverse();
  while(bi<rev.length){
    const nd=rev[bi];nd.bA=true;nd.bD=true;
    if(nd.tp==='o'){
      const grads=lgrads(nd),descs=gfDesc(nd);
      bwdAnn={node:nd,upstream:nd.grad,items:grads.map((gr,i)=>({target:gr.t,local:gr.l,downstream:gr.g,desc:descs[i]||''}))};
      grads.forEach(gr=>gr.t.grad+=gr.g)
    }else bwdAnn=null;
    ph=bi+1>=rev.length?'bwd_done':'backward';
    snap('backward-step',nd.tp==='o'?`Backward ${bi+1}: ${nd.op}`:`Backward ${bi+1}: ${nd.label}`);nd.bA=false;bi++
  }
  return{exportedAt:new Date().toISOString(),expression:expr,variables:vars,stageCount:stages.length,stages}
}
function saveStagesJSON(name=`autograd-stages-${fileStamp()}.json`){
  const tl=stageTimeline();if(!tl)return null;
  downloadText(JSON.stringify(tl,null,2),name,'application/json');return tl
}
function emitStateChange(reason){
  const snap=graphSnapshot();if(!snap)return;
  snap.reason=reason;
  window.dispatchEvent(new CustomEvent('autograd:statechange',{detail:snap}));
  subs.forEach(fn=>{try{fn(snap)}catch(err){console.error(err)}})
}
function phaseMeta(){
  if(!S.g)return{key:'idle',label:'Idle',note:'Build a graph to begin.'};
  const total=S.tp.length||0;
  switch(S.ph){
    case'idle':return{key:'idle',label:'Ready',note:`${total} nodes loaded. Start the forward pass.`};
    case'forward':return{key:'forward',label:'Forward',note:`Resolving node ${Math.min(S.fi+1,total)} of ${total}.`};
    case'fwd_done':return{key:'fwd_done',label:'Forward done',note:`Output = ${fm(S.g.out.val)}. Backprop is ready.`};
    case'backward':return{key:'backward',label:'Backward',note:`Propagating gradient step ${Math.min(S.bi+1,total)} of ${total}.`};
    case'bwd_done':return{key:'bwd_done',label:'Gradients ready',note:'All gradients computed. Export or inspect any variable now.'};
    default:return{key:'idle',label:'Idle',note:'Build a graph to begin.'}
  }
}
function bindAPI(){
  window.AutogradPlayground={
    getState:graphSnapshot,
    snapshot:graphSnapshot,
    timeline:stageTimeline,
    serializeSVG:serializeCurrentSvg,
    saveJSON:saveGraphJSON,
    saveStagesJSON,
    saveSVG:saveGraphSvg,
    build(expr,vars={}){if(typeof expr==='string')$('expr-input').value=expr;buildFrom(vars);return graphSnapshot()},
    loadExample(i){loadEx(i);return graphSnapshot()},
    reset(){resetS();return graphSnapshot()},
    forwardStep(){fwdStep();return graphSnapshot()},
    forwardAll(){fwdAll();return graphSnapshot()},
    backwardStep(){bwdStep();return graphSnapshot()},
    backwardAll(){bwdAll();return graphSnapshot()},
    fitView(){fitGraph();return graphSnapshot()},
    zoomIn(){zoomGraph(1.18);return graphSnapshot()},
    zoomOut(){zoomGraph(1/1.18);return graphSnapshot()},
    focusActive(){focusNode();return graphSnapshot()},
    subscribe(fn){if(typeof fn!=='function')throw new Error('Listener must be a function');subs.add(fn);return()=>subs.delete(fn)},
    unsubscribe(fn){subs.delete(fn);return subs.size}
  }
}
function applyTheme(theme){
  document.body.dataset.theme=theme;
  localStorage.setItem('autograd-theme',theme);
  $('theme-toggle').setAttribute('aria-pressed',theme==='dark'?'true':'false')
}
function initTheme(){
  const stored=localStorage.getItem('autograd-theme');
  const theme=stored||(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  applyTheme(theme)
}
function bindCanvasNav(){
  const svg=$('graph-svg'),wrap=$('graph-wrap');
  svg.addEventListener('wheel',e=>{if(!S.g)return;e.preventDefault();zoomGraph(e.deltaY<0?1.14:1/1.14,clientToGraph(svg,e.clientX,e.clientY))},{passive:false});
  svg.addEventListener('mousedown',e=>{
    if(!S.g||e.button!==0)return;
    S.cam.drag=true;S.cam.last={x:e.clientX,y:e.clientY};S.cam.moved=false;wrap.classList.add('dragging')
  });
  window.addEventListener('mousemove',e=>{
    if(!S.cam.drag||!S.cam.current)return;
    const r=svg.getBoundingClientRect();if(!r.width||!r.height)return;
    const dx=(e.clientX-S.cam.last.x)*S.cam.current.w/r.width,dy=(e.clientY-S.cam.last.y)*S.cam.current.h/r.height;
    if(dx||dy){
      S.cam.current.x-=dx;S.cam.current.y-=dy;S.cam.last={x:e.clientX,y:e.clientY};
      S.cam.manual=true;S.cam.moved=true;clampCamera();requestRender()
    }
  });
  function endDrag(){
    if(!S.cam.drag)return;
    S.cam.drag=false;wrap.classList.remove('dragging');
    if(S.cam.moved)emitStateChange('pan-view')
  }
  window.addEventListener('mouseup',endDrag);
  svg.addEventListener('dblclick',()=>fitGraph());
}

function initEx(){
  const row=$('examples-row'),sel=$('example-select'),groups={};
  row.innerHTML='';sel.innerHTML='';
  EX.forEach((x,i)=>{
    const b=document.createElement('button');b.className='ex-btn';
    b.innerHTML=`${x.n}<span class="ex-desc">${x.d}</span>`;b.title=x.e;
    b.addEventListener('click',()=>loadEx(i));row.appendChild(b);
    if(!groups[x.c]){const g=document.createElement('optgroup');g.label=x.c;groups[x.c]=g;sel.appendChild(g)}
    const opt=document.createElement('option');opt.value=String(i);opt.textContent=`${x.n} · ${x.d}`;groups[x.c].appendChild(opt)
  });
  sel.addEventListener('change',e=>loadEx(parseInt(e.target.value,10)||0))
}

function loadEx(i){document.querySelectorAll('.ex-btn').forEach((b,j)=>b.classList.toggle('active',j===i));
$('example-select').value=String(i);$('expr-input').value=EX[i].e;buildFrom(EX[i].v)}

function buildFrom(dv){
  const expr=$('expr-input').value.trim();if(!expr){info('Type an expression.','');return}
  stopP();
  try{const ast=parse(tokenize(expr));const g=buildG(ast);
    const bar=$('var-bar');bar.innerHTML='';
    g.vo.forEach(nm=>{const ch=document.createElement('div');ch.className='var-chip';
      const sp=document.createElement('span');sp.textContent=nm+' =';ch.appendChild(sp);
      const inp=document.createElement('input');inp.type='number';inp.step='any';inp.value=dv?.[nm]??1;
      inp.dataset.v=nm;inp.addEventListener('change',()=>resetS());ch.appendChild(inp);bar.appendChild(ch)});
    S.g=g;S.tp=topo(g.out);S.dm=layoutG(g);S.cam={base:null,current:null,manual:false,drag:false,last:null,moved:false};resetS(true);
    $('graph-empty').style.display='none';$('graph-svg').style.display='block';$('legend').style.display='flex';
    info('Graph built! Press F or click Fwd to start.','');
    upUI();emitStateChange('build');
  }catch(e){info('Error: '+e.message,'err')}}

function resetS(skipEmit=false){if(!S.g)return;stopP();S.fi=0;S.bi=0;S.ph='idle';S.bwdAnn=null;
  S.g.ns.forEach(n=>{n.fD=false;n.bD=false;n.fA=false;n.bA=false;n.grad=0;
    if(n.tp==='c')n.val=n.val;else if(n.tp==='i')n.val=null;else n.val=null});
  $('var-bar').querySelectorAll('input').forEach(inp=>{const nm=inp.dataset.v;if(S.g.vs[nm])S.g.vs[nm].val=parseFloat(inp.value)||0});
  upUI();renderG($('graph-svg'),S.g,S.dm,null);$('grad-card').style.display='none';$('chain-rule-area').innerHTML='';
  if(!skipEmit)emitStateChange('reset')}

function fwdStep(){
  if(!S.g)return;if(S.ph==='idle')S.ph='forward';if(S.ph!=='forward')return;
  S.g.ns.forEach(n=>n.fA=false);S.bwdAnn=null;
  if(S.fi>=S.tp.length){S.ph='fwd_done';upUI();return}
  const nd=S.tp[S.fi];
  if(nd.tp!=='i'&&nd.tp!=='c')nd.val=compV(nd);
  nd.fD=true;nd.fA=true;
  if(nd.tp==='i')info(`Fwd ${S.fi+1}/${S.tp.length}: Input "${nd.label}" = ${fm(nd.val)}`,'fwd');
  else if(nd.tp==='c')info(`Fwd ${S.fi+1}/${S.tp.length}: Const = ${fm(nd.val)}`,'fwd');
  else info(`Fwd ${S.fi+1}/${S.tp.length}: ${descOp(nd)}\n= ${fm(nd.val)}`,'fwd');
  $('chain-rule-area').innerHTML='';
  S.fi++;if(S.fi>=S.tp.length)S.ph='fwd_done';
  upUI();renderG($('graph-svg'),S.g,S.dm,null);emitStateChange('forward-step')}

function fwdAll(){if(!S.g)return;stopP();S.g.ns.forEach(n=>n.fA=false);S.bwdAnn=null;
  while(S.fi<S.tp.length){const n=S.tp[S.fi];if(n.tp!=='i'&&n.tp!=='c')n.val=compV(n);n.fD=true;S.fi++}
  S.ph='fwd_done';info(`Forward complete. Output = ${fm(S.g.out.val)}\n\nPress B or Bwd to compute gradients.`,'fwd');
  $('chain-rule-area').innerHTML='';upUI();renderG($('graph-svg'),S.g,S.dm,null);emitStateChange('forward-all')}

function bwdStep(){
  if(!S.g)return;
  if(S.ph==='fwd_done'){S.ph='backward';S.g.out.grad=1;S.bi=0}
  if(S.ph!=='backward')return;
  S.g.ns.forEach(n=>n.bA=false);
  const rev=[...S.tp].reverse();
  if(S.bi>=rev.length){S.ph='bwd_done';upUI();return}
  const nd=rev[S.bi];nd.bA=true;nd.bD=true;

  if(nd.tp==='o'){
    const upstream=nd.grad;
    const grads=lgrads(nd),descs=gfDesc(nd);
    const items=grads.map((gr,i)=>({target:gr.t,local:gr.l,downstream:gr.g,desc:descs[i]||''}));

    // Store annotation for SVG rendering
    S.bwdAnn={node:nd,upstream,items};

    // Propagate
    grads.forEach(gr=>gr.t.grad+=gr.g);

    // Info text
    const posLabel=(n,i)=>{const p=nd.ins.length>1?['Left','Right'][i]:'';return`${p} "${n.label}" (=${fm(n.val)})`};
    let txt=`Bwd ${S.bi+1}/${rev.length}: ${nd.op} node\n`;
    txt+=`Accumulated grad = ${fm(upstream)}\n`;
    info(txt,'bwd');

    // Chain rule visual in sidebar
    renderChainRule(nd,upstream,items);
  } else if(nd.tp==='i'){
    S.bwdAnn=null;
    info(`Bwd ${S.bi+1}/${rev.length}: Variable "${nd.label}"\n\u2207 = ${fm(nd.grad)}`,'bwd');
    $('chain-rule-area').innerHTML='';
  } else {
    S.bwdAnn=null;
    info(`Bwd ${S.bi+1}/${rev.length}: Constant ${nd.label}`,'bwd');
    $('chain-rule-area').innerHTML='';
  }
  S.bi++;if(S.bi>=rev.length){S.ph='bwd_done';showGT()}
  upUI();renderG($('graph-svg'),S.g,S.dm,S.bwdAnn);emitStateChange('backward-step')}

function bwdAll(){if(!S.g)return;stopP();
  if(S.ph==='fwd_done'){S.ph='backward';S.g.out.grad=1;S.bi=0}
  if(S.ph!=='backward')return;
  S.g.ns.forEach(n=>n.bA=false);S.bwdAnn=null;
  const rev=[...S.tp].reverse();
  while(S.bi<rev.length){const nd=rev[S.bi];nd.bD=true;
    if(nd.tp==='o')lgrads(nd).forEach(gr=>gr.t.grad+=gr.g);S.bi++}
  S.ph='bwd_done';info('Backward complete! All gradients computed.','bwd');
  $('chain-rule-area').innerHTML='';showGT();upUI();renderG($('graph-svg'),S.g,S.dm,null);emitStateChange('backward-all')}

function renderChainRule(nd,upstream,items){
  const area=$('chain-rule-area');
  let h='<div class="chain-rule"><div style="font-weight:700;margin-bottom:6px;color:#334155">Chain Rule at this node:</div>';
  items.forEach((it,i)=>{
    const label=nd.ins.length>1?['Left','Right'][i]+': ':'';
    h+=`<div class="cr-row">
      <span style="color:#64748b;font-size:10px;min-width:36px">${label}</span>
      <span class="cr-pill cr-upstream">${fm(upstream)}</span>
      <span class="cr-arrow">\u00D7</span>
      <span class="cr-pill cr-local">${fm(it.local)}</span>
      <span class="cr-arrow">=</span>
      <span class="cr-pill cr-downstream">${fm(it.downstream)}</span>
      <span class="cr-arrow">\u2192</span>
      <span style="font-weight:600;font-size:11px">${it.target.label}</span>
    </div>`;
    h+=`<div style="font-size:10px;color:#94a3b8;margin:0 0 4px 40px">${it.desc}</div>`;
  });
  h+=`<div style="margin-top:6px;padding-top:6px;border-top:1px solid #e2e8f0;font-size:10px;color:#94a3b8;font-weight:600">
    <span class="cr-pill cr-upstream">upstream</span> \u00D7
    <span class="cr-pill cr-local">local</span> =
    <span class="cr-pill cr-downstream">downstream</span></div></div>`;
  area.innerHTML=h}

function descOp(nd){
  const nm=i=>{const n=nd.ins[i];return n.tp==='i'?n.label:n.tp==='c'?n.label:fm(n.val)};
  const vl=i=>fm(nd.ins[i].val);
  if(nd.ins.length===2)return`${nm(0)} ${nd.label} ${nm(1)}  =  ${vl(0)} ${nd.label} ${vl(1)}`;
  if(nd.ins.length===1)return`${nd.op}(${nm(0)})  =  ${nd.op}(${vl(0)})`;
  return`${nd.op}(${nd.ins.map((_,i)=>vl(i)).join(', ')})`}

function info(t,tp){const b=$('info-box');b.textContent=t;b.className='info-box'+(tp==='fwd'?' fwd':tp==='bwd'?' bwd':tp==='err'?' err':'')}

function showGT(){const tb=$('grad-table').querySelector('tbody');tb.innerHTML='';
  S.g.vo.forEach(nm=>{const n=S.g.vs[nm];const tr=document.createElement('tr');
    tr.innerHTML=`<td style="font-weight:700">${nm}</td><td class="v-val">${fm(n.val)}</td><td class="v-grad" style="text-align:right">${fm(n.grad)}</td>`;
    tb.appendChild(tr)});$('grad-card').style.display='block'}

function toggleP(){if(pTimer){stopP();return}
  const d=parseInt($('speed').value);$('btn-play').classList.add('playing');$('btn-play').innerHTML='&#9646;&#9646;';
  pTimer=setInterval(()=>{if(S.ph==='idle'||S.ph==='forward')fwdStep();
    else if(S.ph==='fwd_done'||S.ph==='backward')bwdStep();if(S.ph==='bwd_done')stopP()},d)}
function stopP(){if(pTimer){clearInterval(pTimer);pTimer=null}$('btn-play').classList.remove('playing');$('btn-play').innerHTML='&#9654;'}

function upUI(){const p=S.ph,h=!!S.g;
  $('btn-fwd-step').disabled=!h||(p!=='idle'&&p!=='forward');$('btn-fwd-all').disabled=!h||(p!=='idle'&&p!=='forward');
  $('btn-bwd-step').disabled=!h||(p!=='fwd_done'&&p!=='backward');$('btn-bwd-all').disabled=!h||(p!=='fwd_done'&&p!=='backward');
  $('btn-reset').disabled=!h;$('btn-play').disabled=!h||p==='bwd_done';
  $('btn-export-json').disabled=!h;$('btn-export-stages').disabled=!h;$('btn-export-svg').disabled=!h;
  const pf=$('pbar'),total=S.tp.length||1;
  if(p==='idle'){pf.style.width='0%';pf.className='progress-fill'}
  else if(p==='forward'||p==='fwd_done'){pf.style.width=`${(S.fi/total)*50}%`;pf.className='progress-fill fwd'}
  else{pf.style.width=`${50+(S.bi/total)*50}%`;pf.className='progress-fill bwd'}
  const meta=phaseMeta(),pill=$('phase-pill');
  pill.textContent=meta.label;
  pill.className='status-pill'+(meta.key&&meta.key!=='idle'?' '+meta.key:'');
  $('status-note').textContent=meta.note}

// =====================================================================
//  EVENTS
// =====================================================================
$('btn-build').addEventListener('click',()=>buildFrom());
$('btn-reset').addEventListener('click',resetS);
$('btn-fwd-step').addEventListener('click',fwdStep);
$('btn-fwd-all').addEventListener('click',fwdAll);
$('btn-bwd-step').addEventListener('click',bwdStep);
$('btn-bwd-all').addEventListener('click',bwdAll);
$('btn-play').addEventListener('click',toggleP);
$('btn-zoom-in').addEventListener('click',()=>zoomGraph(1.18,null,true));
$('btn-zoom-out').addEventListener('click',()=>zoomGraph(1/1.18,null,true));
$('btn-fit').addEventListener('click',()=>fitGraph(true));
$('btn-focus').addEventListener('click',()=>focusNode());
$('btn-export-json').addEventListener('click',()=>saveGraphJSON());
$('btn-export-stages').addEventListener('click',()=>saveStagesJSON());
$('btn-export-svg').addEventListener('click',()=>saveGraphSvg());
$('theme-toggle').addEventListener('click',()=>applyTheme(document.body.dataset.theme==='dark'?'light':'dark'));
$('expr-input').addEventListener('keydown',e=>{if(e.key==='Enter')buildFrom()});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key==='f'||e.key==='F')fwdStep();if(e.key==='b'||e.key==='B')bwdStep();
  if(e.key==='r'||e.key==='R')resetS();if(e.key===' '){e.preventDefault();toggleP()}
  if(e.key==='+'||e.key==='=')zoomGraph(1.18,null,true);if(e.key==='-')zoomGraph(1/1.18,null,true);if(e.key==='0')fitGraph(true)});

// =====================================================================
//  INIT
// =====================================================================
initTheme();bindAPI();bindCanvasNav();initEx();upUI();loadEx(0);
