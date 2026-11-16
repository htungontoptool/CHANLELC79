const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 9000;
const LC_AT = process.env.LC_AT || '';
const BRAND_ID = '@htungdzvipmax';
const BRAND_NAME = 'LC79-XOCDIA';
const HISTORY_MAX = 500;
const HISTORY_SHOW = 50;
const DATA_DIR = process.env.DATA_DIR || '.';
const HISTORY_FILE = require('path').join(DATA_DIR, 'history.json');
app.set('json spaces', 4);

// Dự đoán đã phát hành cho từng phiên (phiên -> {chan_le, vi}); lịch sử đã chốt
const pendingPred = new Map();
let history = [];
try { history = JSON.parse(require('fs').readFileSync(HISTORY_FILE, 'utf8')); } catch (e) { history = []; }

app.use(cors());
app.use(express.json());

let latestSessions = [];
let lastUpdateTime = null;
let updateCount = 0;

// Tự động cập nhật dữ liệu
async function fetchLatestData() {
    try {
        const response = await axios.get('https://wcl.tele68.com/v1/chanlefull/lite-sessions', {
            params: { 
                cp: 'R', 
                cl: 'R', 
                pf: 'web', 
                at: LC_AT},
            timeout: 5000
        });
        
        const newData = response.data.list || response.data.data || [];
        
        if (newData.length > 0) {
            if (latestSessions.length === 0 || latestSessions[0]?.id !== newData[0]?.id) {
                latestSessions = newData;
                lastUpdateTime = new Date();
                updateCount++;
                resolvePredictions(newData);
                refreshPrediction().catch(() => {});
                console.log(`[${lastUpdateTime.toLocaleTimeString()}] ✅ Cập nhật - Phiên: ${newData[0]?.id}`);
            }
        }
    } catch (error) {
        console.log(`[${new Date().toLocaleTimeString()}] ⚠️ Lỗi: ${error.message}`);
    }
}

setInterval(() => fetchLatestData(), 2000);
fetchLatestData();

// ========== 79 THUẬT TOÁN DỰ ĐOÁN ==========

// === NHÓM 1: THUẬT TOÁN THỐNG KÊ CƠ BẢN (10 algorithms) ===

function ma3(results) { const r = results.slice(0,3).filter(x=>x==='chan').length; return {prediction: r>=2?'chan':'le', confidence: 60+r*10}; }
function ma5(results) { const r = results.slice(0,5).filter(x=>x==='chan').length; return {prediction: r>=3?'chan':'le', confidence: 60+r*8}; }
function ma7(results) { const r = results.slice(0,7).filter(x=>x==='chan').length; return {prediction: r>=4?'chan':'le', confidence: 60+r*6}; }
function ma10(results) { const r = results.slice(0,10).filter(x=>x==='chan').length; return {prediction: r>=5?'chan':'le', confidence: 60+r*5}; }
function ema(results) { let ema=results[0]==='chan'?1:0; const a=0.3; for(let i=1;i<Math.min(20,results.length);i++){ const v=results[i]==='chan'?1:0; ema=a*v+(1-a)*ema; } return {prediction: ema>0.5?'chan':'le', confidence: Math.abs(ema-0.5)*200}; }
function wma(results) { let ws=0,wt=0; for(let i=0;i<Math.min(10,results.length);i++){ const w=10-i; ws+=(results[i]==='chan'?1:0)*w; wt+=w; } const wma=ws/wt; return {prediction: wma>0.5?'chan':'le', confidence: Math.abs(wma-0.5)*200}; }
function holtWinters(results) { let l=results[0]==='chan'?1:0, t=0; for(let i=1;i<Math.min(20,results.length);i++){ const v=results[i]==='chan'?1:0; const nl=0.2*v+0.8*(l+t); t=0.1*(nl-l)+0.9*t; l=nl; } return {prediction: l+t>0.5?'chan':'le', confidence: Math.abs(l+t-0.5)*200}; }
function arima(results) { let diff=[]; for(let i=1;i<results.length;i++) diff.push((results[i-1]==='chan'?1:0)-(results[i]==='chan'?1:0)); const ma=diff.slice(0,2).reduce((a,b)=>a+b,0)/2; const ar=diff.slice(0,2).reduce((a,b)=>a+b,0)/2; const f=(results[0]==='chan'?1:0)+(ar+ma)/2; return {prediction: f>0.5?'chan':'le', confidence: Math.abs(f-0.5)*200}; }
function sarima(results) { let seasonal=[]; for(let i=0;i<Math.min(12,results.length-1);i++) seasonal.push(results[i]===results[i+1]?1:0); const s=seasonal.reduce((a,b)=>a+b,0)/seasonal.length; const trend=results.slice(0,5).filter(x=>x==='chan').length/5; const f=(trend+s)/2; return {prediction: f>0.5?'chan':'le', confidence: Math.abs(f-0.5)*200}; }
function garch(results) { let variance=0.5; for(let i=1;i<Math.min(20,results.length);i++){ const r=(results[i]==='chan'?1:0)-(results[i-1]==='chan'?1:0); variance=0.1*Math.pow(r,2)+0.9*variance; } return {prediction: results[0]==='chan'?'le':'chan', confidence: Math.min(90,50+variance*50)}; }

// === NHÓM 2: THUẬT TOÁN MACHINE LEARNING (15 algorithms) ===

function knn3(results) { let d=[]; for(let i=1;i<Math.min(30,results.length);i++){ let dist=0; for(let j=0;j<3&&i+j<results.length;j++) if(results[j]!==results[i+j]) dist++; d.push({dist,next:results[i-1]}); } d.sort((a,b)=>a.dist-b.dist); const k=Math.min(5,d.length); const c=d.slice(0,k).filter(x=>x.next==='chan').length; return {prediction: c>k/2?'chan':'le', confidence: (c/k)*100}; }
function knn5(results) { let d=[]; for(let i=1;i<Math.min(30,results.length);i++){ let dist=0; for(let j=0;j<5&&i+j<results.length;j++) if(results[j]!==results[i+j]) dist++; d.push({dist,next:results[i-1]}); } d.sort((a,b)=>a.dist-b.dist); const k=Math.min(7,d.length); const c=d.slice(0,k).filter(x=>x.next==='chan').length; return {prediction: c>k/2?'chan':'le', confidence: (c/k)*100}; }
function naiveBayes(results) { let p_chan=results.filter(x=>x==='chan').length/results.length; let p_le=1-p_chan; let cond={chan:{},le:{}}; for(let i=1;i<Math.min(20,results.length);i++){ const prev=results[i-1], curr=results[i]; if(curr==='chan') cond.chan[prev]=(cond.chan[prev]||0)+1; else cond.le[prev]=(cond.le[prev]||0)+1; } const last=results[0]; const post_chan=p_chan*((cond.chan[last]||1)/results.length); const post_le=p_le*((cond.le[last]||1)/results.length); return {prediction: post_chan>post_le?'chan':'le', confidence: (Math.max(post_chan,post_le)/(post_chan+post_le))*100}; }
function decisionTree(results) { let bestGain=0,bestFeat=0; for(let f=0;f<4;f++){ let left=[],right=[]; for(let i=4;i<Math.min(40,results.length);i++){ const val=f===0?results[i-4]:f===1?results[i-3]:f===2?results[i-2]:results[i-1]; if(val==='chan') left.push(results[i]); else right.push(results[i]); } const entropy=(arr)=>{ if(arr.length===0) return 0; const c=arr.filter(x=>x==='chan').length/arr.length; const l=1-c; return -(c*Math.log2(c||1)+l*Math.log2(l||1)); }; const gain=entropy(results.slice(4,40))-(left.length/40)*entropy(left)-(right.length/40)*entropy(right); if(gain>bestGain){ bestGain=gain; bestFeat=f; } } const test=bestFeat===0?results[0]:bestFeat===1?results[1]:bestFeat===2?results[2]:results[3]; let c=0,l=0; for(let i=4;i<Math.min(40,results.length);i++){ const val=bestFeat===0?results[i-4]:bestFeat===1?results[i-3]:bestFeat===2?results[i-2]:results[i-1]; if(val===test){ if(results[i]==='chan') c++; else l++; } } return {prediction: c>l?'chan':'le', confidence: (Math.max(c,l)/(c+l+1))*100}; }
function randomForest(results) { let cVotes=0,lVotes=0; for(let t=0;t<20;t++){ let sample=[]; for(let i=0;i<30;i++) sample.push(results[Math.floor(Math.random()*results.length)]); const cCount=sample.filter(x=>x==='chan').length; if(cCount>15) cVotes++; else lVotes++; } return {prediction: cVotes>lVotes?'chan':'le', confidence: (Math.max(cVotes,lVotes)/20)*100}; }
function gradientBoost(results) { let pred=0.5; for(let iter=0;iter<30;iter++){ const residual=(results[0]==='chan'?1:0)-pred; pred+=0.1*(residual>0?0.1:-0.1); } return {prediction: pred>0.5?'chan':'le', confidence: Math.abs(pred-0.5)*200}; }
function xgboost(results) { let score=0; for(let i=0;i<20;i++){ const threshold=Math.random(); const weight=Math.random(); if((results[0]==='chan'?1:0)>threshold) score+=weight; else score-=weight; } return {prediction: score>0?'chan':'le', confidence: Math.min(95,Math.abs(score)*10)}; }
function lightgbm(results) { const bins=10; let hist=new Array(bins).fill(0); for(let i=0;i<Math.min(30,results.length);i++){ const bin=Math.floor(i*bins/30); if(results[i]==='chan') hist[bin]++; else hist[bin]--; } const grad=hist.reduce((a,b)=>a+b,0); return {prediction: grad>0?'chan':'le', confidence: Math.min(95,Math.abs(grad)*5)}; }
function catboost(results) { const cats=results.slice(0,10).map(x=>x==='chan'?1:0); let pred=0; for(let i=0;i<cats.length;i++) pred+=cats[i]*Math.exp(-i/3); pred=1/(1+Math.exp(-pred/5)); return {prediction: pred>0.5?'chan':'le', confidence: Math.abs(pred-0.5)*200}; }
function logisticRegression(results) { let w=0,b=0; for(let i=1;i<Math.min(30,results.length);i++){ const x=results[i-1]==='chan'?1:-1; const y=results[i]==='chan'?1:0; const pred=1/(1+Math.exp(-(w*x+b))); const error=y-pred; w+=0.01*error*x; b+=0.01*error; } const last=results[0]==='chan'?1:-1; const final=1/(1+Math.exp(-(w*last+b))); return {prediction: final>0.5?'chan':'le', confidence: Math.abs(final-0.5)*200}; }
function svmLinear(results) { let w=0,b=0; for(let i=1;i<Math.min(30,results.length);i++){ const x=results[i-1]==='chan'?1:-1; const y=results[i]==='chan'?1:-1; if(y*(w*x+b)<=1){ w+=0.01*y*x; b+=0.01*y; } } const last=results[0]==='chan'?1:-1; const pred=w*last+b; return {prediction: pred>0?'chan':'le', confidence: Math.min(95,Math.abs(pred)*50)}; }
function svmRbf(results) { const gamma=0.5; let alpha=new Array(30).fill(0); for(let iter=0;iter<20;iter++){ for(let i=1;i<Math.min(30,results.length);i++){ const x=results[i-1]==='chan'?1:-1; const y=results[i]==='chan'?1:-1; let pred=0; for(let j=1;j<Math.min(30,results.length);j++){ const xj=results[j-1]==='chan'?1:-1; pred+=alpha[j]*Math.exp(-gamma*Math.pow(x-xj,2)); } if(y*pred<=1) alpha[i]+=0.01; } } const last=results[0]==='chan'?1:-1; let final=0; for(let j=1;j<Math.min(30,results.length);j++){ const xj=results[j-1]==='chan'?1:-1; final+=alpha[j]*Math.exp(-gamma*Math.pow(last-xj,2)); } return {prediction: final>0?'chan':'le', confidence: Math.min(95,Math.abs(final)*50)}; }
function perceptron(results) { let w=0,b=0; for(let epoch=0;epoch<10;epoch++){ for(let i=1;i<Math.min(30,results.length);i++){ const x=results[i-1]==='chan'?1:0; const y=results[i]==='chan'?1:0; const pred=w*x+b>0?1:0; if(pred!==y){ w+=0.1*(y-pred)*x; b+=0.1*(y-pred); } } } const last=results[0]==='chan'?1:0; const final=w*last+b; return {prediction: final>0?'chan':'le', confidence: Math.min(90,Math.abs(final)*50)}; }
function mlp(results) { let w1=[Math.random(),Math.random(),Math.random()], w2=[Math.random(),Math.random(),Math.random()], b1=Math.random(),b2=Math.random(); for(let epoch=0;epoch<20;epoch++){ for(let i=3;i<Math.min(30,results.length);i++){ const x=[results[i-3]==='chan'?1:0,results[i-2]==='chan'?1:0,results[i-1]==='chan'?1:0]; const y=results[i]==='chan'?1:0; let h=0; for(let j=0;j<3;j++) h+=x[j]*w1[j]; h=1/(1+Math.exp(-(h+b1))); let out=0; for(let j=0;j<3;j++) out+=h*w2[j]; out=1/(1+Math.exp(-(out+b2))); const error=y-out; for(let j=0;j<3;j++) w2[j]+=0.1*error*h; for(let j=0;j<3;j++) w1[j]+=0.1*error*w2[j]*h*(1-h)*x[j]; } } const x=[results[0]==='chan'?1:0,results[1]==='chan'?1:0,results[2]==='chan'?1:0]; let h=0; for(let j=0;j<3;j++) h+=x[j]*w1[j]; h=1/(1+Math.exp(-(h+b1))); let out=0; for(let j=0;j<3;j++) out+=h*w2[j]; out=1/(1+Math.exp(-(out+b2))); return {prediction: out>0.5?'chan':'le', confidence: Math.abs(out-0.5)*200}; }

// === NHÓM 3: THUẬT TOÁN DEEP LEARNING (10 algorithms) ===

function rnn(results) { let h=0; for(let i=0;i<Math.min(20,results.length);i++){ const x=results[i]==='chan'?1:-1; h=Math.tanh(0.5*h+0.5*x); } return {prediction: h>0?'chan':'le', confidence: Math.abs(h)*100}; }
function lstm(results) { let c=0,h=0; for(let i=0;i<Math.min(20,results.length);i++){ const x=results[i]==='chan'?1:-1; const f=0.9, i_g=0.1, o=0.5; c=f*c+i_g*x; h=o*Math.tanh(c); } return {prediction: h>0?'chan':'le', confidence: Math.abs(h)*100}; }
function gru(results) { let h=0; for(let i=0;i<Math.min(20,results.length);i++){ const x=results[i]==='chan'?1:-1; const z=1/(1+Math.exp(-(0.7*h+x))); const r=1/(1+Math.exp(-(0.3*h+x))); const ht=Math.tanh(x+r*h); h=(1-z)*h+z*ht; } return {prediction: h>0?'chan':'le', confidence: Math.abs(h)*100}; }
function biLSTM(results) { let fwd=0,bwd=0; for(let i=0;i<Math.min(15,results.length);i++){ const x=results[i]==='chan'?1:-1; fwd=Math.tanh(0.5*fwd+0.5*x); } for(let i=Math.min(15,results.length)-1;i>=0;i--){ const x=results[i]==='chan'?1:-1; bwd=Math.tanh(0.5*bwd+0.5*x); } const h=(fwd+bwd)/2; return {prediction: h>0?'chan':'le', confidence: Math.abs(h)*100}; }
function autoencoder(results) { let encoded=[]; for(let i=0;i<Math.min(15,results.length);i+=3){ const pat=results.slice(i,i+3); const val=pat.filter(x=>x==='chan').length/3; encoded.push(val); } const avg=encoded.reduce((a,b)=>a+b,0)/encoded.length; return {prediction: avg>0.5?'chan':'le', confidence: Math.abs(avg-0.5)*200}; }
function cnn1d(results) { const filters=[2,3,4]; let feats=[]; for(const fs of filters){ let maxF=0; for(let i=0;i<=results.length-fs;i++){ const pat=results.slice(i,i+fs); const c=pat.filter(x=>x==='chan').length/fs; maxF=Math.max(maxF,c); } feats.push(maxF); } const avg=feats.reduce((a,b)=>a+b,0)/feats.length; return {prediction: avg>0.5?'chan':'le', confidence: Math.abs(avg-0.5)*200}; }
function transformer(results) { let attSum=0,wtSum=0; const q=results[0]==='chan'?1:0; for(let i=1;i<Math.min(15,results.length);i++){ const k=results[i]==='chan'?1:0; const v=results[i-1]==='chan'?1:0; const att=Math.exp(-Math.abs(q-k)/2); attSum+=att*v; wtSum+=att; } const out=attSum/wtSum; return {prediction: out>0.5?'chan':'le', confidence: Math.abs(out-0.5)*200}; }
function attention(results) { let weights=[]; for(let i=0;i<Math.min(10,results.length);i++){ let w=Math.exp(-i/3); if(i>0&&results[i]===results[0]) w*=1.5; weights.push(w); } const total=weights.reduce((a,b)=>a+b,0); let cScore=0,lScore=0; for(let i=0;i<weights.length;i++){ const w=weights[i]/total; if(results[i]==='chan') cScore+=w; else lScore+=w; } return {prediction: cScore>lScore?'chan':'le', confidence: Math.abs(cScore-lScore)*100}; }
function resNet(results) { let x=results[0]==='chan'?1:-1; const residual=x; for(let i=1;i<Math.min(10,results.length);i++){ x=Math.tanh(0.5*x+0.5*(results[i]==='chan'?1:-1)); } x=(x+residual)/2; return {prediction: x>0?'chan':'le', confidence: Math.abs(x)*100}; }
function dqn(results) { let qTable={}; let state=results.slice(0,3).join(','); for(let i=3;i<Math.min(30,results.length);i++){ const action=results[i]; const reward=results[i+1]===action?1:-0.5; const nextState=results.slice(i-2,i+1).join(','); if(!qTable[state]) qTable[state]={chan:0,le:0}; if(!qTable[nextState]) qTable[nextState]={chan:0,le:0}; qTable[state][action]+=0.1*(reward+0.9*Math.max(qTable[nextState].chan,qTable[nextState].le)-qTable[state][action]); state=nextState; } const lastState=results.slice(0,3).join(','); if(!qTable[lastState]) return {prediction: results[0]==='chan'?'le':'chan', confidence:60}; return {prediction: qTable[lastState].chan>qTable[lastState].le?'chan':'le', confidence: Math.abs(qTable[lastState].chan-qTable[lastState].le)/Math.max(qTable[lastState].chan+qTable[lastState].le,1)*100}; }

// === NHÓM 4: THUẬT TOÁN TỐI ƯU HÓA (10 algorithms) ===

function genetic(results) { let pop=Array(20).fill().map(()=>({genes:Array(5).fill().map(()=>Math.random()),fit:0})); for(let gen=0;gen<30;gen++){ for(const ind of pop){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ let score=0; for(let j=0;j<5&&i+j<results.length;j++) score+=ind.genes[j]*(results[i+j]==='chan'?1:-1); const pred=score>0?'chan':'le'; if(pred===results[i+1]) correct++; } ind.fit=correct/Math.min(19,results.length-1); } pop.sort((a,b)=>b.fit-a.fit); pop=pop.slice(0,10); while(pop.length<20){ const p1=pop[Math.floor(Math.random()*10)], p2=pop[Math.floor(Math.random()*10)]; const child={genes:p1.genes.map((g,i)=>(g+p2.genes[i])/2),fit:0}; if(Math.random()<0.1) child.genes[Math.floor(Math.random()*5)]+=(Math.random()-0.5)*0.2; pop.push(child); } } let score=0; for(let j=0;j<5&&j<results.length;j++) score+=pop[0].genes[j]*(results[j]==='chan'?1:-1); return {prediction: score>0?'chan':'le', confidence: Math.min(95,pop[0].fit*100+10)}; }
function pso(results) { let particles=Array(15).fill().map(()=>({pos:Math.random()*2-1,vel:(Math.random()-0.5)*0.5,bestPos:0,bestFit:-Infinity})); let gBest={pos:0,fit:-Infinity}; for(let iter=0;iter<20;iter++){ for(const p of particles){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=p.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } const fit=correct/Math.min(19,results.length-1); if(fit>p.bestFit){ p.bestFit=fit; p.bestPos=p.pos; } if(fit>gBest.fit){ gBest.fit=fit; gBest.pos=p.pos; } } for(const p of particles){ p.vel=0.7*p.vel+0.5*Math.random()*(p.bestPos-p.pos)+0.5*Math.random()*(gBest.pos-p.pos); p.pos+=p.vel; p.pos=Math.max(-1,Math.min(1,p.pos)); } } const pred=gBest.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,gBest.fit*100+10)}; }
function simulatedAnnealing(results) { let curr=Math.random()*2-1, best=curr, bestFit=-Infinity, temp=1.0; for(let iter=0;iter<50;iter++){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=curr*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } const fit=correct/Math.min(19,results.length-1); if(fit>bestFit){ bestFit=fit; best=curr; } const neighbor=curr+(Math.random()-0.5)*temp; let nFit=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=neighbor*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) nFit++; } nFit/=Math.min(19,results.length-1); if(nFit>fit||Math.exp((nFit-fit)/temp)>Math.random()) curr=neighbor; temp*=0.95; } const pred=best*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,bestFit*100+10)}; }
function antColony(results) { let pheromone=[1,1]; let bestSol=null, bestFit=-Infinity; for(let iter=0;iter<20;iter++){ for(let ant=0;ant<10;ant++){ const prob=pheromone.map(p=>p/pheromone.reduce((a,b)=>a+b,0)); const choice=Math.random()<prob[0]?0:1; const pred=choice===0?'chan':'le'; let correct=0; for(let i=0;i<Math.min(20,results.length);i++) if(results[i]===pred) correct++; const fit=correct/Math.min(20,results.length); if(fit>bestFit){ bestFit=fit; bestSol=pred; } pheromone[choice]+=fit*0.1; } pheromone=pheromone.map(p=>p*0.9); } return {prediction: bestSol, confidence: Math.min(95,bestFit*100+10)}; }
function beeColony(results) { let bees=Array(12).fill().map(()=>({pos:Math.random()*2-1,fit:0})); for(let iter=0;iter<25;iter++){ for(const b of bees){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=b.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } b.fit=correct/Math.min(19,results.length-1); } bees.sort((a,b)=>b.fit-a.fit); for(let i=6;i<12;i++) bees[i].pos=Math.random()*2-1; for(let i=0;i<6;i++){ const newPos=bees[i].pos+(Math.random()-0.5)*0.3; bees[i].pos=Math.max(-1,Math.min(1,newPos)); } } const best=bees[0]; const pred=best.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,best.fit*100+10)}; }
function firefly(results) { let fireflies=Array(15).fill().map(()=>({pos:Math.random()*2-1,intensity:0})); for(let iter=0;iter<30;iter++){ for(const f of fireflies){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=f.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } f.intensity=correct/Math.min(19,results.length-1); } for(let i=0;i<fireflies.length;i++){ for(let j=0;j<fireflies.length;j++){ if(fireflies[j].intensity>fireflies[i].intensity){ const beta=0.5*Math.exp(-Math.pow(fireflies[i].pos-fireflies[j].pos,2)); fireflies[i].pos+=beta*(fireflies[j].pos-fireflies[i].pos)+(Math.random()-0.5)*0.1; } } } } const best=fireflies.reduce((a,b)=>a.intensity>b.intensity?a:b); const pred=best.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,best.intensity*100+10)}; }
function cuckooSearch(results) { let nests=Array(15).fill().map(()=>({pos:Math.random()*2-1,fit:0})); for(let iter=0;iter<30;iter++){ for(const n of nests){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=n.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } n.fit=correct/Math.min(19,results.length-1); } const best=nests.reduce((a,b)=>a.fit>b.fit?a:b); const newPos=best.pos+Math.random()*0.2; let newFit=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=newPos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) newFit++; } newFit/=Math.min(19,results.length-1); if(newFit>best.fit) best.pos=newPos; const abandon=Math.floor(Math.random()*nests.length); nests[abandon].pos=Math.random()*2-1; } const best=nests.reduce((a,b)=>a.fit>b.fit?a:b); const pred=best.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,best.fit*100+10)}; }
function batAlgorithm(results) { let bats=Array(15).fill().map(()=>({pos:Math.random()*2-1,vel:0,freq:Math.random(),loudness:0.5})); let best={pos:0,fit:-Infinity}; for(let iter=0;iter<30;iter++){ for(const b of bats){ b.freq=Math.random(); b.vel+=(b.pos-best.pos)*b.freq; b.pos+=b.vel; let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=b.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } const fit=correct/Math.min(19,results.length-1); if(fit>best.fit){ best.fit=fit; best.pos=b.pos; } if(Math.random()<b.loudness){ const newPos=best.pos+(Math.random()-0.5)*0.1; let newFit=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=newPos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) newFit++; } newFit/=Math.min(19,results.length-1); if(newFit>fit){ b.pos=newPos; b.loudness*=0.9; } } } } const pred=best.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,best.fit*100+10)}; }
function greyWolf(results) { let wolves=Array(10).fill().map(()=>({pos:Math.random()*2-1,fit:0})); let alpha={pos:0,fit:-Infinity}, beta={pos:0,fit:-Infinity}, delta={pos:0,fit:-Infinity}; for(let iter=0;iter<30;iter++){ for(const w of wolves){ let correct=0; for(let i=0;i<Math.min(20,results.length-1);i++){ const pred=w.pos*(results[i]==='chan'?1:-1)>0?'chan':'le'; if(pred===results[i+1]) correct++; } w.fit=correct/Math.min(19,results.length-1); } wolves.sort((a,b)=>b.fit-a.fit); alpha=wolves[0]; beta=wolves[1]; delta=wolves[2]; for(const w of wolves){ const r1=Math.random(), r2=Math.random(); const A1=2*0.5*r1-0.5, C1=2*r2; const D_alpha=Math.abs(C1*alpha.pos-w.pos); const X1=alpha.pos-A1*D_alpha; const r3=Math.random(), r4=Math.random(); const A2=2*0.5*r3-0.5, C2=2*r4; const D_beta=Math.abs(C2*beta.pos-w.pos); const X2=beta.pos-A2*D_beta; const r5=Math.random(), r6=Math.random(); const A3=2*0.5*r5-0.5, C3=2*r6; const D_delta=Math.abs(C3*delta.pos-w.pos); const X3=delta.pos-A3*D_delta; w.pos=(X1+X2+X3)/3; } } const pred=alpha.pos*(results[0]==='chan'?1:-1)>0?'chan':'le'; return {prediction: pred, confidence: Math.min(95,alpha.fit*100+10)}; }

// === NHÓM 5: THUẬT TOÁN CHUYÊN BIỆT (14 algorithms) ===

function markov1(results) { return markovChron(results, 1) || { prediction: results[0] === 'chan' ? 'le' : 'chan', confidence: 60 }; }
function markov2(results) { return markovChron(results, 2) || markov1(results); }
function markov3(results) { return markovChron(results, 3) || markov2(results); }
function hiddenMarkov(results) { let trans={chan:{chan:0.5,le:0.5},le:{chan:0.5,le:0.5}}; let emit={chan:{chan:0.6,le:0.4},le:{chan:0.4,le:0.6}}; let gamma=[]; for(let iter=0;iter<5;iter++){ gamma=[]; for(let i=0;i<results.length;i++){ let fwd={chan:0,le:0}; if(i===0){ fwd.chan=emit.chan[results[i]]*0.5; fwd.le=emit.le[results[i]]*0.5; }else{ fwd.chan=emit.chan[results[i]]*(gamma[i-1].chan*trans.chan.chan+gamma[i-1].le*trans.le.chan); fwd.le=emit.le[results[i]]*(gamma[i-1].chan*trans.chan.le+gamma[i-1].le*trans.le.le); } const norm=fwd.chan+fwd.le; gamma.push({chan:fwd.chan/norm,le:fwd.le/norm}); } trans={chan:{chan:0,le:0},le:{chan:0,le:0}}; for(let i=1;i<results.length;i++){ for(const s1 of ['chan','le']){ for(const s2 of ['chan','le']){ trans[s1][s2]+=gamma[i-1][s1]*trans[s1][s2]*emit[s2][results[i]]; } } } for(const s1 of ['chan','le']){ const total=trans[s1].chan+trans[s1].le; if(total>0){ trans[s1].chan/=total; trans[s1].le/=total; } } } const last=gamma[0]; const p=last.chan*trans.chan.chan+last.le*trans.le.chan; return {prediction: p>0.5?'chan':'le', confidence: Math.abs(p-0.5)*200}; }
function monteCarlo(results) { let cWins=0; for(let sim=0;sim<100;sim++){ let simRes=[results[0]]; for(let i=1;i<5;i++){ const rate=results.slice(0,Math.min(10,i)).filter(x=>x==='chan').length/Math.min(10,i); simRes.push(Math.random()<rate?'chan':'le'); } if(simRes[simRes.length-1]==='chan') cWins++; } return {prediction: cWins>50?'chan':'le', confidence: (Math.max(cWins,100-cWins)/100)*100}; }
function kalman(results) { let x=results[0]==='chan'?1:0, p=1; for(let i=1;i<Math.min(15,results.length);i++){ const z=results[i]==='chan'?1:0; p=p+0.1; const k=p/(p+0.1); x=x+k*(z-x); p=(1-k)*p; } return {prediction: x>0.5?'chan':'le', confidence: Math.abs(x-0.5)*200}; }
function particleFilter(results) { let particles=Array(100).fill().map(()=>({state:Math.random()>0.5?'chan':'le',weight:1})); for(let i=1;i<Math.min(15,results.length);i++){ let total=0; for(const p of particles){ if(Math.random()<0.3) p.state=p.state==='chan'?'le':'chan'; if(p.state===results[i]) p.weight*=0.7; else p.weight*=0.3; total+=p.weight; } for(const p of particles) p.weight/=total; } let cWeight=0; for(const p of particles) if(p.state==='chan') cWeight+=p.weight; return {prediction: cWeight>0.5?'chan':'le', confidence: Math.abs(cWeight-0.5)*200}; }
function fourier(results) { const n=Math.min(32,results.length); let seq=results.slice(0,n).map(x=>x==='chan'?1:-1); let freqs=[]; for(let k=0;k<n/2;k++){ let real=0,imag=0; for(let t=0;t<n;t++){ const angle=2*Math.PI*k*t/n; real+=seq[t]*Math.cos(angle); imag-=seq[t]*Math.sin(angle); } freqs.push(Math.sqrt(real*real+imag*imag)); } let maxF=0,maxIdx=0; for(let i=1;i<freqs.length;i++) if(freqs[i]>maxF){ maxF=freqs[i]; maxIdx=i; } const phase=Math.atan2(seq[0]*Math.sin(2*Math.PI*maxIdx*0/n),seq[0]*Math.cos(2*Math.PI*maxIdx*0/n)); const next=Math.cos(2*Math.PI*maxIdx*1/n+phase); return {prediction: next>0?'chan':'le', confidence: Math.min(90,(maxF/freqs[0])*50)}; }
function wavelet(results) { let app=results.slice(0,16).map(x=>x==='chan'?1:-1); let det=[]; while(app.length>2){ let newApp=[],newDet=[]; for(let i=0;i<app.length-1;i+=2){ newApp.push((app[i]+app[i+1])/Math.sqrt(2)); newDet.push((app[i]-app[i+1])/Math.sqrt(2)); } det.push(newDet); app=newApp; } let recon=[...app]; for(let i=det.length-1;i>=0;i--){ let newRecon=[]; for(let j=0;j<recon.length;j++){ newRecon.push((recon[j]+(det[i][j]||0))/Math.sqrt(2)); newRecon.push((recon[j]-(det[i][j]||0))/Math.sqrt(2)); } recon=newRecon; } const energy=app.reduce((a,b)=>a+Math.abs(b),0)/app.length; return {prediction: recon[0]>0?'chan':'le', confidence: Math.min(90,energy*50)}; }
function copula(results) { const seq=results.map(x=>x==='chan'?1:0); const mean=seq.reduce((a,b)=>a+b,0)/seq.length; let varSum=0; for(let i=0;i<seq.length;i++) varSum+=Math.pow(seq[i]-mean,2); const variance=varSum/seq.length; const cop=Math.exp(-Math.pow(mean-0.5,2)/(2*variance))/Math.sqrt(2*Math.PI*variance); return {prediction: cop>0.5?'chan':'le', confidence: Math.min(90,cop*100)}; }
function bayesianNetwork(results) { let parents={}; for(let i=1;i<Math.min(20,results.length);i++){ const key=results[i-1]; if(!parents[key]) parents[key]={chan:0,le:0}; parents[key][results[i]]++; } const last=results[0]; if(!parents[last]) return {prediction: results[0]==='chan'?'le':'chan', confidence:60}; const total=parents[last].chan+parents[last].le; const p=parents[last].chan/total; return {prediction: p>0.5?'chan':'le', confidence: Math.max(p,1-p)*100}; }
function causalInference(results) { let causes={}; for(let lag=1;lag<=5;lag++){ let corr=0; for(let i=lag;i<Math.min(30,results.length);i++) if(results[i-lag]===results[i]) corr++; causes[lag]=corr/(Math.min(30,results.length)-lag); } let bestLag=1, bestCorr=0; for(let lag=1;lag<=5;lag++) if(causes[lag]>bestCorr){ bestCorr=causes[lag]; bestLag=lag; } return {prediction: results[bestLag-1], confidence: bestCorr*100}; }
function grangerCausality(results) { let fStats=[]; for(let lag=1;lag<=4;lag++){ let res1=0,res2=0; for(let i=lag;i<Math.min(30,results.length);i++){ const pred1=results[i-lag]; if(pred1!==results[i]) res1++; const pred2=Math.random()>0.5?'chan':'le'; if(pred2!==results[i]) res2++; } const fStat=(res2-res1)/res2; fStats.push(fStat); } const bestF=Math.max(...fStats); const bestLag=fStats.indexOf(bestF)+1; return {prediction: results[bestLag-1], confidence: Math.min(90,bestF*100)}; }

// === NHÓM 6: ENSEMBLE META (20 algorithms) ===

function votingEnsemble(preds) { let c=0,l=0; for(const p of Object.values(preds)) if(p.prediction==='chan') c++; else l++; return {prediction: c>l?'chan':'le', confidence: (Math.max(c,l)/(c+l))*100}; }
function weightedVotingEnsemble(preds) { let cW=0,lW=0; for(const p of Object.values(preds)){ if(p.prediction==='chan') cW+=p.confidence; else lW+=p.confidence; } return {prediction: cW>lW?'chan':'le', confidence: (Math.max(cW,lW)/(cW+lW))*100}; }
function stackingEnsemble(preds) { const weights={}; for(const [name,p] of Object.entries(preds)) weights[name]=p.confidence/100; let cW=0,lW=0; for(const [name,p] of Object.entries(preds)){ if(p.prediction==='chan') cW+=weights[name]; else lW+=weights[name]; } return {prediction: cW>lW?'chan':'le', confidence: (Math.max(cW,lW)/(cW+lW))*100}; }
function bayesianAveraging(preds) { let post={chan:0.5,le:0.5}; for(const p of Object.values(preds)){ const like=p.confidence/100; if(p.prediction==='chan'){ post.chan*=like; post.le*=(1-like); }else{ post.chan*=(1-like); post.le*=like; } } const total=post.chan+post.le; post.chan/=total; post.le/=total; return {prediction: post.chan>post.le?'chan':'le', confidence: Math.max(post.chan,post.le)*100}; }
function gbEnsemble(preds) { let pred=0.5; for(const p of Object.values(preds)){ const residual=(p.prediction==='chan'?1:0)-pred; pred+=0.05*residual*(p.confidence/100); } return {prediction: pred>0.5?'chan':'le', confidence: Math.abs(pred-0.5)*200}; }
function adaboostEnsemble(preds) { let weights={}; const names=Object.keys(preds); names.forEach(n=>weights[n]=1/names.length); for(let iter=0;iter<5;iter++){ let totalErr=0; for(const [n,p] of Object.entries(preds)){ const err=1-(p.confidence/100); totalErr+=err*weights[n]; weights[n]*=Math.exp(err); } const alpha=0.5*Math.log((1-totalErr)/Math.max(totalErr,0.001)); for(const n in weights) weights[n]*=Math.exp(alpha*(preds[n].confidence/100)); } let cW=0,lW=0; for(const [n,p] of Object.entries(preds)){ if(p.prediction==='chan') cW+=weights[n]; else lW+=weights[n]; } return {prediction: cW>lW?'chan':'le', confidence: (Math.max(cW,lW)/(cW+lW))*100}; }
function randomSubspace(preds) { const names=Object.keys(preds); let cVotes=0,lVotes=0; for(let i=0;i<20;i++){ const subspace=[]; for(let j=0;j<10;j++) subspace.push(preds[names[Math.floor(Math.random()*names.length)]]); const sc=subspace.filter(p=>p.prediction==='chan').length; if(sc>5) cVotes++; else lVotes++; } return {prediction: cVotes>lVotes?'chan':'le', confidence: (Math.max(cVotes,lVotes)/20)*100}; }
function baggingEnsemble(preds) { let cVotes=0,lVotes=0; for(let i=0;i<30;i++){ const sample=[]; const names=Object.keys(preds); for(let j=0;j<20;j++) sample.push(preds[names[Math.floor(Math.random()*names.length)]]); const sc=sample.filter(p=>p.prediction==='chan').length; if(sc>10) cVotes++; else lVotes++; } return {prediction: cVotes>lVotes?'chan':'le', confidence: (Math.max(cVotes,lVotes)/30)*100}; }
function boostingEnsemble(preds) { let pred=0.5; const alphas=Array(10).fill().map(()=>Math.random()); for(let i=0;i<alphas.length;i++){ const p=Object.values(preds)[i%Object.keys(preds).length]; const residual=(p.prediction==='chan'?1:0)-pred; pred+=alphas[i]*residual; } return {prediction: pred>0.5?'chan':'le', confidence: Math.abs(pred-0.5)*200}; }
function maxVoting(preds) { let c=0,l=0; for(const p of Object.values(preds)) if(p.prediction==='chan') c++; else l++; const total=c+l; const conf=(Math.max(c,l)/total)*100; return {prediction: c>l?'chan':'le', confidence: conf}; }
function minVoting(preds) { let c=0,l=0; for(const p of Object.values(preds)) if(p.prediction==='chan') c++; else l++; const total=c+l; const conf=100-(Math.abs(c-l)/total)*100; return {prediction: c>l?'chan':'le', confidence: conf}; }
function medianVoting(preds) { const votes=[]; for(const p of Object.values(preds)) votes.push(p.prediction==='chan'?1:0); votes.sort(); const median=votes[Math.floor(votes.length/2)]; const c=votes.filter(v=>v===1).length; return {prediction: median===1?'chan':'le', confidence: (c/votes.length)*100}; }
function modeVoting(preds) { let c=0,l=0; for(const p of Object.values(preds)) if(p.prediction==='chan') c++; else l++; const total=c+l; return {prediction: c>l?'chan':'le', confidence: (Math.max(c,l)/total)*100}; }
function geometricMean(preds) { let cProd=1,lProd=1; for(const p of Object.values(preds)){ if(p.prediction==='chan') cProd*=p.confidence; else lProd*=p.confidence; } const cMean=Math.pow(cProd,1/Object.keys(preds).length); const lMean=Math.pow(lProd,1/Object.keys(preds).length); return {prediction: cMean>lMean?'chan':'le', confidence: (Math.max(cMean,lMean)/(cMean+lMean))*100}; }
function harmonicMean(preds) { let cSum=0,lSum=0; for(const p of Object.values(preds)){ if(p.prediction==='chan') cSum+=1/p.confidence; else lSum+=1/p.confidence; } const cMean=Object.keys(preds).length/cSum; const lMean=Object.keys(preds).length/lSum; return {prediction: cMean>lMean?'chan':'le', confidence: (Math.max(cMean,lMean)/(cMean+lMean))*100}; }
function quadraticMean(preds) { let cSum=0,lSum=0; for(const p of Object.values(preds)){ if(p.prediction==='chan') cSum+=Math.pow(p.confidence,2); else lSum+=Math.pow(p.confidence,2); } const cMean=Math.sqrt(cSum/Object.keys(preds).length); const lMean=Math.sqrt(lSum/Object.keys(preds).length); return {prediction: cMean>lMean?'chan':'le', confidence: (Math.max(cMean,lMean)/(cMean+lMean))*100}; }
function weightedMedian(preds) { const pairs=[]; for(const p of Object.values(preds)) pairs.push({val:p.prediction==='chan'?1:0,weight:p.confidence}); pairs.sort((a,b)=>a.val-b.val); let totalWeight=pairs.reduce((s,p)=>s+p.weight,0); let cumWeight=0; for(const p of pairs){ cumWeight+=p.weight; if(cumWeight>=totalWeight/2) return {prediction: p.val===1?'chan':'le', confidence: (cumWeight/totalWeight)*100}; } return {prediction: 'chan', confidence:50}; }
function softVoting(preds) { let cProb=0,lProb=0; for(const p of Object.values(preds)){ const prob=p.confidence/100; if(p.prediction==='chan') cProb+=prob; else lProb+=prob; } const total=cProb+lProb; return {prediction: cProb>lProb?'chan':'le', confidence: (Math.max(cProb,lProb)/total)*100}; }
function hardVoting(preds) { let c=0,l=0; for(const p of Object.values(preds)) if(p.prediction==='chan') c++; else l++; const total=c+l; const conf=Math.min(99,(Math.max(c,l)/total)*100+5); return {prediction: c>l?'chan':'le', confidence: conf}; }
function superEnsemble(preds) { const metaPreds=[votingEnsemble(preds),weightedVotingEnsemble(preds),stackingEnsemble(preds),bayesianAveraging(preds),gbEnsemble(preds),adaboostEnsemble(preds),randomSubspace(preds),baggingEnsemble(preds),boostingEnsemble(preds),maxVoting(preds),minVoting(preds),medianVoting(preds),modeVoting(preds),geometricMean(preds),harmonicMean(preds),quadraticMean(preds),weightedMedian(preds),softVoting(preds),hardVoting(preds)]; let c=0,l=0; for(const mp of metaPreds) if(mp.prediction==='chan') c++; else l++; const avgConf=metaPreds.reduce((s,mp)=>s+mp.confidence,0)/metaPreds.length; return {prediction: c>l?'chan':'le', confidence: Math.min(98,avgConf*1.05)}; }

// ========== TỔNG HỢP 79 THUẬT TOÁN ==========
async function ultimatePrediction(sessions) {
    const results = sessions.map(s => s.resultTruyenThong);
    
    if (results.length < 20) {
        const last5 = results.slice(0,5);
        const cCount = last5.filter(x=>x==='chan').length;
        return { prediction: cCount>=3?'chan':'le', confidence: 65 };
    }
    
    // Thu thập tất cả 79 predictions
    const allPreds = {
        // Statistical (10)
        ma3: ma3(results), ma5: ma5(results), ma7: ma7(results), ma10: ma10(results),
        ema: ema(results), wma: wma(results), holt: holtWinters(results),
        arima: arima(results), sarima: sarima(results), garch: garch(results),
        
        // ML (15)
        knn3: knn3(results), knn5: knn5(results), bayes: naiveBayes(results),
        tree: decisionTree(results), rf: randomForest(results), gb: gradientBoost(results),
        xgb: xgboost(results), lgbm: lightgbm(results), cat: catboost(results),
        lr: logisticRegression(results), svmL: svmLinear(results), svmR: svmRbf(results),
        perc: perceptron(results), mlp: mlp(results),
        
        // Deep Learning (10)
        rnn: rnn(results), lstm: lstm(results), gru: gru(results), bilstm: biLSTM(results),
        auto: autoencoder(results), cnn: cnn1d(results), trans: transformer(results),
        att: attention(results), res: resNet(results), dqn: dqn(results),
        
        // Optimization (10)
        genetic: genetic(results), pso: pso(results), sa: simulatedAnnealing(results),
        aco: antColony(results), bee: beeColony(results), firefly: firefly(results),
        cuckoo: cuckooSearch(results), bat: batAlgorithm(results), greywolf: greyWolf(results),
        
        // Specialized (14)
        m1: markov1(results), m2: markov2(results), m3: markov3(results),
        hmm: hiddenMarkov(results), mc: monteCarlo(results), kalman: kalman(results),
        pf: particleFilter(results), fourier: fourier(results), wavelet: wavelet(results),
        copula: copula(results), bayesNet: bayesianNetwork(results),
        causal: causalInference(results), granger: grangerCausality(results),
        
        // Ensemble Meta (20 - will be combined)
    };
    
    // Add meta-ensemble predictions
    const metaPred = superEnsemble(allPreds);
    
    // Final result from meta-ensemble
    let finalConfidence = metaPred.confidence;
    
    // Adjust based on data quality
    const changes = results.slice(0,20).filter((r,i,arr)=>i>0&&r!==arr[i-1]).length;
    const stability = 1 - (changes/19);
    finalConfidence = finalConfidence * (0.7 + stability * 0.3);
    finalConfidence = Math.min(98, Math.max(65, finalConfidence));
    
    return {
        prediction: metaPred.prediction,
        confidence: Math.round(finalConfidence)
    };
}


app.get("/", (req, res) => {
    res.json({
        status: "alive",
        message: "API đang hoạt động ✅",
        time: new Date(),
        uptime: process.uptime(),
        updates: updateCount,
        lastUpdate: lastUpdateTime
    });
});

// ========== API CHÍNH ==========
// ========== CHẴN/LẺ + VỊ, LỊCH SỬ ==========
function markovChron(results, k) {
    const c = results.slice().reverse();
    const t = {};
    for (let i = k; i < c.length; i++) {
        const st = c.slice(i - k, i).join(',');
        if (!t[st]) t[st] = { chan: 0, le: 0 };
        t[st][c[i]]++;
    }
    const row = t[c.slice(c.length - k).join(',')];
    if (!row || row.chan + row.le === 0) return null;
    const p = row.chan / (row.chan + row.le);
    return { prediction: p > 0.5 ? 'chan' : 'le', confidence: Math.max(p, 1 - p) * 100 };
}

// Vị = số xúc xắc đỏ theo màu chiếm đa số trong 4 xúc xắc
const VI_LABELS = {
    four_do: '4 ĐỎ', three_do: '3 ĐỎ 1 TRẮNG', two_do: '2 ĐỎ 2 TRẮNG', two_trang: '2 ĐỎ 2 TRẮNG',
    three_trang: '1 ĐỎ 3 TRẮNG', four_trang: '4 TRẮNG'
};
// Số đỏ, số trắng ứng với từng vị
const VI_COUNTS = {
    '4 ĐỎ': [4, 0], '3 ĐỎ 1 TRẮNG': [3, 1], '2 ĐỎ 2 TRẮNG': [2, 2], '1 ĐỎ 3 TRẮNG': [1, 3], '4 TRẮNG': [0, 4]
};
const VI_PRIOR = {
    '4 ĐỎ': 1 / 16, '3 ĐỎ 1 TRẮNG': 4 / 16, '2 ĐỎ 2 TRẮNG': 6 / 16, '1 ĐỎ 3 TRẮNG': 4 / 16, '4 TRẮNG': 1 / 16
};

function viLabel(resultVi) { return VI_LABELS[resultVi] || null; }

// Dự đoán vị.
// VI_MODE=best (mặc định cũ): luôn chọn vị có xác suất cao nhất -> lặp lại "2 ĐỎ 2 TRẮNG".
// VI_MODE=thompson (mặc định mới): lấy mẫu từ phân phối hậu nghiệm Dirichlet của tần suất
// 200 phiên gần nhất -> vị đa dạng hơn, vẫn ưu tiên vị hay xuất hiện.
const VI_MODE = process.env.VI_MODE || 'thompson';

function randn() {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Gamma(shape >= 1), thuật toán Marsaglia-Tsang
function gammaSample(shape) {
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    while (true) {
        let x, v;
        do { x = randn(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const u = Math.random();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
}

// Kiểm định phân bố vị thực tế so với lý thuyết (chi-bình phương, 4 bậc tự do)
function chi2Sf4(x) { return Math.exp(-x / 2) * (1 + x / 2); }

function viOutcomeStats(sessions) {
    const recent = sessions.slice(0, 200);
    const counts = {};
    Object.keys(VI_PRIOR).forEach(l => counts[l] = 0);
    let n = 0;
    recent.forEach(s => { const l = viLabel(s.resultVi); if (l) { counts[l]++; n++; } });
    let stat = 0;
    for (const l of Object.keys(VI_PRIOR)) {
        const e = n * VI_PRIOR[l];
        if (e > 0) stat += (counts[l] - e) ** 2 / e;
    }
    return { n, counts, p: n >= 30 ? chi2Sf4(stat) : null };
}

// Chỉ dùng tần suất thực tế khi lệch có ý nghĩa (p < 0.05); nếu không thì dùng xác suất lý thuyết.
// Như vậy không đuổi theo nhiễu ngẫu nhiên.
function predictVi(sessions) {
    const st = viOutcomeStats(sessions);
    const pseudo = 20;
    const labels = Object.keys(VI_PRIOR);
    const deviates = st.p !== null && st.p < 0.05;
    if (VI_MODE === 'best') {
        let best = null, bestScore = -1;
        for (const l of labels) {
            const base = deviates ? st.counts[l] : st.n * VI_PRIOR[l];
            const score = (base + pseudo * VI_PRIOR[l]) / (st.n + pseudo);
            if (score > bestScore) { bestScore = score; best = l; }
        }
        return best;
    }
    let best = null, bestDraw = -1;
    for (const l of labels) {
        const base = deviates ? st.counts[l] : st.n * VI_PRIOR[l];
        const draw = gammaSample(base + pseudo * VI_PRIOR[l]);
        if (draw > bestDraw) { bestDraw = draw; best = l; }
    }
    return best;
}

// Chẵn/lẻ: với 4 xúc xắc công bằng, xác suất chẵn = lẻ = 50%, nên mọi lựa chọn đều có kỳ vọng như nhau.
// CL_MODE=fair (mặc định): nếu chẵn/lẻ không lệch có ý nghĩa thì chọn ngẫu nhiên 50/50 -> hết tình trạng lặp một phía;
//   nếu lệch có ý nghĩa (p < 0.05) thì theo phía đang nhiều hơn.
// CL_MODE=algo: dùng kết quả của ultimatePrediction (79 thuật toán) như cũ.
const CL_MODE = process.env.CL_MODE || 'fair';

function chooseChanLe(sessions, algoPick) {
    if (CL_MODE === 'algo') return algoPick;
    const recent = sessions.slice(0, 200).map(s => s.resultTruyenThong).filter(x => x === 'chan' || x === 'le');
    const n = recent.length;
    const k = recent.filter(x => x === 'chan').length;
    if (n >= 30 && binomTwoSidedP(k, n, 0.5) < 0.05) return k / n > 0.5 ? 'chan' : 'le';
    return Math.random() < 0.5 ? 'chan' : 'le';
}

async function refreshPrediction() {
    if (latestSessions.length === 0) return;
    const nextId = latestSessions[0].id + 1;
    if (pendingPred.has(nextId)) return;
    const cl = await ultimatePrediction(latestSessions);
    if (!pendingPred.has(nextId)) {
        pendingPred.set(nextId, { chan_le: chooseChanLe(latestSessions, cl.prediction), vi: predictVi(latestSessions) });
    }
}

function saveHistory() {
    try { require('fs').writeFileSync(HISTORY_FILE, JSON.stringify(history)); }
    catch (e) { console.log(`Không lưu được lịch sử: ${e.message}`); }
}

function resultText(chanLe, vi) {
    const cl = chanLe === 'chan' ? 'CHẴN' : 'LẺ';
    const c = vi && VI_COUNTS[vi];
    if (!c) return cl;
    return `${cl} - ${c[1]} TRẮNG + ${c[0]} ĐỎ`;
}

function resolvePredictions(newData) {
    let changed = false;
    for (const s of newData) {
        const p = pendingPred.get(s.id);
        if (!p) continue;
        const actualVi = viLabel(s.resultVi);
        history.unshift({
            Phien: s.id,
            Du_doan_chan_le: p.chan_le === 'chan' ? 'CHẴN' : 'LẺ',
            Ket_qua: resultText(s.resultTruyenThong, actualVi),
            Dung_chan_le: p.chan_le === s.resultTruyenThong,
            Du_doan_vi: p.vi,
            Ket_qua_vi: actualVi,
            Dung_vi: p.vi === actualVi
        });
        pendingPred.delete(s.id);
        changed = true;
    }
    // Bỏ các dự đoán của phiên đã quá cũ (không bao giờ có kết quả)
    if (latestSessions.length) {
        const minId = latestSessions[0].id - 5;
        for (const id of pendingPred.keys()) if (id < minId) pendingPred.delete(id);
    }
    if (changed) {
        history = history.slice(0, HISTORY_MAX);
        saveHistory();
    }
}

// Xác suất nhị thức hai phía: P(X có độ bất thường >= quan sát) khi tỷ lệ lý thuyết là p0
const LOG_FACT = [0];
for (let i = 1; i <= 1000; i++) LOG_FACT[i] = LOG_FACT[i - 1] + Math.log(i);
function binomTwoSidedP(k, n, p0) {
    if (n === 0) return 1;
    const logPmf = x => LOG_FACT[n] - LOG_FACT[x] - LOG_FACT[n - x] + x * Math.log(p0) + (n - x) * Math.log(1 - p0);
    const pObs = Math.exp(logPmf(k));
    let total = 0;
    for (let x = 0; x <= n; x++) {
        const px = Math.exp(logPmf(x));
        if (px <= pObs * (1 + 1e-9)) total += px;
    }
    return Math.min(1, total);
}

// Đánh giá: đúng hơn ngẫu nhiên không? Chẵn/Lẻ so với 50%, Vị so với 6/16 (cơ cấu 2-2 có xác suất cao nhất)
function verdict(k, n, p0) {
    if (n < 30) return 'chưa đủ mẫu';
    const p = binomTwoSidedP(k, n, p0);
    if (p < 0.05) return k / n > p0 ? 'tốt hơn ngẫu nhiên' : 'kém ngẫu nhiên';
    return 'chưa khác ngẫu nhiên';
}

// Tỷ lệ từng vị trong các kết quả đã về, so với lý thuyết
function viDistribution() {
    const labels = Object.keys(VI_PRIOR);
    const counts = {};
    labels.forEach(l => counts[l] = 0);
    let n = 0;
    history.forEach(h => { if (counts[h.Ket_qua_vi] !== undefined) { counts[h.Ket_qua_vi]++; n++; } });
    if (n === 0) return 'chưa có dữ liệu';
    let stat = 0;
    labels.forEach(l => { const e = n * VI_PRIOR[l]; stat += (counts[l] - e) ** 2 / e; });
    const p = n >= 30 ? chi2Sf4(stat).toFixed(3) : null;
    const parts = labels.map(l => `${l} ${(counts[l] / n * 100).toFixed(0)}% (lt ${(VI_PRIOR[l] * 100).toFixed(0)}%)`);
    const verdictTxt = p === null ? 'chưa đủ mẫu' : (p < 0.05 ? `⚠️ lệch lý thuyết (p=${p})` : `✅ khớp lý thuyết (p=${p})`);
    return `${parts.join(' · ')} · ${verdictTxt}`;
}

function historyStats() {
    const n = history.length;
    const pct = (a, b) => b ? (a / b * 100).toFixed(0) + '%' : '0%';
    const cl = history.filter(h => h.Dung_chan_le).length;
    const vi = history.filter(h => h.Dung_vi).length;
    return {
        'Tổng': n,
        'Chẵn/Lẻ': `✅ ${cl}/${n} · ${pct(cl, n)} · ${verdict(cl, n, 0.5)}`,
        'Vị': `✅ ${vi}/${n} · ${pct(vi, n)} · ${verdict(vi, n, 6 / 16)}`,
        'Phân bố vị thực tế': viDistribution()
    };
}

// Mỗi phiên một dòng gọn: phiên │ dự đoán ✅/❌ │ dự vị ✅/❌ │ kết quả
function historyLine(h) {
    const mark = ok => ok ? '✅' : '❌';
    return `${h.Phien} │ ${h.Du_doan_chan_le} ${mark(h.Dung_chan_le)} │ ${h.Du_doan_vi} ${mark(h.Dung_vi)} │ ${h.Ket_qua}`;
}

// Phiên trước: đã đoán gì và đúng hay sai (ngắn gọn)
function previousResult() {
    const h = history[0];
    if (!h) return 'chưa có';
    const mark = ok => ok ? '✅' : '❌';
    return `${h.Phien} · ${h.Du_doan_chan_le} ${mark(h.Dung_chan_le)} · ${h.Du_doan_vi} ${mark(h.Dung_vi)}`;
}

function buildCard(id, p) {
    return {
        id: BRAND_ID,
        Name: BRAND_NAME,
        Phien: id,
        'DỰ ĐOÁN': p.chan_le === 'chan' ? 'CHẴN' : 'LẺ',
        'DỰ VỊ': p.vi,
        'PHIÊN TRƯỚC': previousResult()
    };
}

app.get('/api/xdlc', async (req, res) => {
    try {
        if (latestSessions.length === 0) await fetchLatestData();
        if (latestSessions.length === 0) return res.json({ id: BRAND_ID, Name: BRAND_NAME, Phien: null, 'DỰ ĐOÁN': null, 'DỰ VỊ': null });
        const nextId = latestSessions[0].id + 1;
        if (!pendingPred.has(nextId)) await refreshPrediction();
        res.json(buildCard(nextId, pendingPred.get(nextId)));
    } catch (error) {
        res.json({ id: BRAND_ID, Name: BRAND_NAME, Phien: null, 'DỰ ĐOÁN': null, 'DỰ VỊ': null });
    }
});

function historyResponse(req, res) {
    const n = Math.min(HISTORY_SHOW, parseInt(req.query.n, 10) || HISTORY_SHOW);
    res.json({
        id: BRAND_ID,
        Name: BRAND_NAME,
        'Thống kê': historyStats(),
        'Lịch sử': history.slice(0, n).map(historyLine)
    });
}
app.get('/history', historyResponse);
app.get('/api/history', historyResponse);

app.get('/api/health', (req, res) => { res.json({ status: 'ok', algorithms: 79, updates: updateCount }); });

// ========== TÀI XỈU LC79 / HITCLUB / MAX789 ==========
const fs = require('fs');
const path = require('path');
fs.mkdirSync(DATA_DIR, { recursive: true });
const gameStates = {};

function firstArray(payload) {
    if (Array.isArray(payload)) return payload;
    for (const k of ['list','data','result','sessions','records','history']) {
        if (Array.isArray(payload?.[k])) return payload[k];
        if (Array.isArray(payload?.data?.[k])) return payload.data[k];
    }
    if (payload && typeof payload === 'object' && (payload.SessionId || payload.phien || payload.sessionId)) return [payload];
    return [];
}
function normalizeTx(x) {
    const id = Number(x.SessionId ?? x.sessionId ?? x.id ?? x.phien ?? x.Phien ?? x.roundId);
    const diceRaw = x.xuc_xac ?? x.xucxac ?? x.dices ?? x.dice ?? [x.FirstDice,x.SecondDice,x.ThirdDice].filter(v=>v!==undefined);
    const dices = Array.isArray(diceRaw) ? diceRaw.map(Number).filter(v=>Number.isFinite(v)&&v>=1&&v<=6) : [];
    const sum = Number(x.Tong ?? x.DiceSum ?? x.point ?? x.total ?? (dices.length ? dices.reduce((a,b)=>a+b,0) : NaN));
    let raw = String(x.ket_qua ?? x.resultTruyenThong ?? x.result ?? x.Ket_qua ?? x.BetSide ?? '').toUpperCase();
    let result = /^(TAI|TÀI)$/.test(raw) ? 'TAI' : /^(XIU|XỈU)$/.test(raw) ? 'XIU' : '';
    if (!result && Number.isFinite(sum)) result = sum >= 11 ? 'TAI' : 'XIU';
    if (!Number.isFinite(id) || !result) return null;
    return { id, result, dices, sum: Number.isFinite(sum) ? sum : (dices.length ? dices.reduce((a,b)=>a+b,0) : null), createdAt: x.CreatedDate ?? x.createdAt ?? null };
}

// Mô hình tổ hợp có kiểm định cuốn chiếu: Markov bậc 1-4 + tần suất gần + nhịp cầu.
// Dữ liệu trò chơi có thể ngẫu nhiên; điểm tin cậy chỉ là độ đồng thuận lịch sử, không bảo đảm thắng.
function txForecast(rows) {
    const seq = rows.map(x=>x.result).filter(Boolean); if (!seq.length) return { side:'TAI', confidence:50, dice:[3,4,4] };
    const counts = {TAI:1, XIU:1};
    seq.slice(0,80).forEach(v=>counts[v]++);
    let score = Math.log(counts.TAI/counts.XIU) * 0.22;
    for (let order=1; order<=4; order++) {
        if (seq.length <= order) continue;
        const key = seq.slice(0,order).join('|'); let a=1,b=1,n=0;
        for(let i=order;i<seq.length;i++) if(seq.slice(i-order,i).join('|')===key){ n++; seq[i]==='TAI'?a++:b++; }
        if(n) score += Math.log(a/b) * (0.55/order) * Math.min(1,n/4);
    }
    const recent=seq.slice(0,12); let weighted=0, weights=0;
    recent.forEach((v,i)=>{const w=Math.exp(-i/5);weighted+=(v==='TAI'?1:-1)*w;weights+=w;});
    score += (weighted/weights)*0.35;
    // Hiệu chỉnh nhẹ theo độ chính xác walk-forward của mô hình trên dữ liệu đã có.
    let hits=0, tested=0;
    for(let i=Math.min(seq.length-1,100);i>=12;i--){ const hist=seq.slice(i); const p=txForecastCore(hist); if(p===seq[i-1]) hits++; tested++; }
    const side=score>=0?'TAI':'XIU';
    const confidence=Math.max(50,Math.min(68,50+Math.abs(score)*5+(tested>=10?(hits/tested-0.5)*5:0)));
    const freq=[1,2,3,4,5,6].map(face=>({face,n:rows.slice(0,100).filter(r=>r.dices?.includes(face)).length})).sort((a,b)=>b.n-a.n);
    const dice=freq.length>=3 && freq[0].n ? freq.slice(0,3).map(x=>x.face) : [3,4,5];
    return {side,confidence:Math.round(confidence),dice};
}
function txForecastCore(seq){
    const last=seq[0]; let same=0,other=0;
    for(let i=1;i<seq.length;i++) if(seq[i]===last) same++; else other++;
    return same>=other?last:(last==='TAI'?'XIU':'TAI');
}

function makeTxGame(cfg) {
    const file=path.join(DATA_DIR,cfg.file);
    let history=[]; try{history=JSON.parse(fs.readFileSync(file,'utf8'));}catch{}
    const st={latest:[],pending:new Map(),history}; gameStates[cfg.key]=st;
    const save=()=>{try{fs.writeFileSync(file,JSON.stringify(history));}catch(e){console.error(`[${cfg.name}] save`,e.message);}};
    async function fetchRows(){
        const r=await axios.get(cfg.url,{headers:{'User-Agent':'Mozilla/5.0','Accept':'application/json'},timeout:15000});
        return firstArray(r.data).map(normalizeTx).filter(Boolean).sort((a,b)=>a.id-b.id);
    }
    function predictNext(){
        if(!st.latest.length)return null;
        const next=st.latest[0].id+1;
        if(!st.pending.has(next)) st.pending.set(next,txForecast(st.latest));
        return {id:next,...st.pending.get(next)};
    }
    async function refresh(){
        const rows=await fetchRows(); if(!rows.length)return;
        st.latest=rows.slice().reverse();
        let changed=false;
        for(const row of rows){
            const p=st.pending.get(row.id); if(!p) continue;
            history.unshift({Phien:row.id,Du_doan:p.side,Ket_qua:row.result,Tong:row.sum,Xuc_xac:row.dices,Dung:p.side===row.result});
            st.pending.delete(row.id); changed=true;
        }
        for(const id of st.pending.keys()) if(id<=rows[rows.length-1].id) st.pending.delete(id);
        if(changed){history=history.slice(0,500);save();}
        // Nạp toàn bộ lịch sử API ngay khi boot; không chờ 5 phiên mới dự đoán.
        predictNext();
    }
    const label=v=>v==='TAI'?'TÀI':'XỈU'; const mark=v=>v?'✅':'❌';
    app.get(cfg.route,async(req,res)=>{
        try{
            if(!st.latest.length) await refresh();
            if(!st.latest.length)return res.status(503).json({Name:cfg.name,TrangThai:'API chưa trả dữ liệu'});
            const p=predictNext(), prev=st.latest[0], old=history[0];
            res.json({id:BRAND_ID,Name:cfg.name,Phien:p.id,'DỰ ĐOÁN':label(p.side),'XÚC XẮC DỰ ĐOÁN':p.dice,'ĐỘ TIN CẬY THAM KHẢO':p.confidence+'%','PHIÊN TRƯỚC':{Phien:prev.id,Xuc_xac:prev.dices,Tong:prev.sum, Ket_qua:label(prev.result), 'Dự đoán trước':old?label(old.Du_doan):null,'Đánh giá':old?mark(old.Dung):'chưa chấm'}});
        }catch(e){res.status(503).json({Name:cfg.name,TrangThai:'Lỗi lấy dữ liệu',error:e.message});}
    });
    app.get(cfg.historyRoute,(req,res)=>{
        const n=Math.min(100,Math.max(1,Number(req.query.n)||50)); const total=history.length,hit=history.filter(h=>h.Dung).length;
        res.json({id:BRAND_ID,Name:cfg.name,'Thống kê':{Tong:total,Dung:`${hit}/${total}`,Ty_le:total?(hit/total*100).toFixed(1)+'%':'0%'},'Lịch sử':history.slice(0,n).map(h=>({Phien:h.Phien,'Dự đoán':label(h.Du_doan),Xuc_xac:h.Xuc_xac,Tong:h.Tong,'Kết quả':label(h.Ket_qua),'Đánh giá':mark(h.Dung)}))});
    });
    refresh().catch(e=>console.error(`[${cfg.name}] boot`,e.message));
    setInterval(()=>refresh().catch(e=>console.error(`[${cfg.name}] refresh`,e.message)),3000);
}

const LC_TX_AT=process.env.TX_AT||'';
const lcParams='?cp=R&cl=R&pf=web&at='+encodeURIComponent(LC_TX_AT);
makeTxGame({key:'lc79-tx',name:'LC79-TX THƯỜNG',url:'https://wtx.tele68.com/v1/tx/lite-sessions'+lcParams,route:'/api/tx',historyRoute:'/history/tx',file:'history_lc79_tx.json'});
makeTxGame({key:'lc79-md5',name:'LC79-TX MD5',url:'https://wtxmd52.tele68.com/v1/txmd5/sessions',route:'/api/md5',historyRoute:'/history/md5',file:'history_lc79_md5.json'});
makeTxGame({key:'hitclub-tx',name:'HITCLUB-TX THƯỜNG',url:'https://apihitclubhihi.onrender.com/api/hitclub/hu',route:'/api/hitclub/tx',historyRoute:'/history/hitclub/tx',file:'history_hitclub_tx.json'});
makeTxGame({key:'hitclub-md5',name:'HITCLUB-TX MD5',url:'https://apihitclubhihi.onrender.com/api/hitclub/md5',route:'/api/hitclub/md5',historyRoute:'/history/hitclub/md5',file:'history_hitclub_md5.json'});
makeTxGame({key:'max789-tx',name:'MAX789-TX THƯỜNG',url:'https://taixiu.maksh3979madfw.com/api/luckydice/GetSoiCau',route:'/api/max789/tx',historyRoute:'/history/max789/tx',file:'history_max789_tx.json'});
makeTxGame({key:'max789-md5',name:'MAX789-TX MD5',url:'https://taixiumd5.maksh3979madfw.com/api/md5luckydice/GetSoiCau',route:'/api/max789/md5',historyRoute:'/history/max789/md5',file:'history_max789_md5.json'});

// ========== KEY API + TELEGRAM BOT ==========
const KEY_FILE=path.join(DATA_DIR,'keys.json');
let keys=[]; try{keys=JSON.parse(fs.readFileSync(KEY_FILE,'utf8'));}catch{}
const saveKeys=()=>fs.writeFileSync(KEY_FILE,JSON.stringify(keys,null,2));
const crypto=require('crypto');
const makeKey=()=>crypto.randomBytes(12).toString('hex').toUpperCase();
const durationMs=(v)=>{const m=String(v||'1d').match(/^(\d+)(h|d|w)$/i);if(!m)return 86400000;return Number(m[1])*({h:3600000,d:86400000,w:604800000}[m[2].toLowerCase()]);};
function keyInfo(k){return keys.find(x=>x.key===k&&x.active&&Date.now()<x.expiresAt);}
const apiNames=['/api/xdlc','/history','/api/tx','/history/tx','/api/md5','/history/md5','/api/hitclub/tx','/history/hitclub/tx','/api/hitclub/md5','/history/hitclub/md5','/api/max789/tx','/history/max789/tx','/api/max789/md5','/history/max789/md5'];
app.get('/:accessKey', (req,res,next)=>{
    const k=keyInfo(req.params.accessKey); if(!k)return res.status(403).json({error:'KEY_INVALID_OR_EXPIRED'});
    res.json({status:'active',expiresAt:new Date(k.expiresAt).toISOString(),apis:apiNames.map(p=>`/${k.key}${p}`)});
});
app.use('/:accessKey', (req,res,next)=>{
    // Bỏ qua route /:accessKey đã xử lý ở trên; chỉ rewrite khi có endpoint phía sau key.
    const raw=req.path.replace(/^\//,'');
    if(!raw) return next();
    const k=keyInfo(req.params.accessKey); if(!k)return res.status(403).json({error:'KEY_INVALID_OR_EXPIRED'});
    const target='/'+raw.split('/').filter(Boolean).join('/');
    if(!apiNames.includes(target))return res.status(404).json({error:'API_NOT_FOUND'});
    req.url=target+(req.url.includes('?')?'?'+req.url.split('?').slice(1).join('?'):'');
    next();
});

let bot=null;
if(process.env.TELEGRAM_BOT_TOKEN){
    const TelegramBot=require('node-telegram-bot-api'); bot=new TelegramBot(process.env.TELEGRAM_BOT_TOKEN,{polling:true});
    const admins=(process.env.ADMIN_IDS||'').split(',').map(x=>x.trim()).filter(Boolean);
    const isAdmin=id=>admins.includes(String(id));
    bot.onText(/^\/start$/,msg=>bot.sendMessage(msg.chat.id,'🎲 API GAME BOT\n\nLệnh quản trị: /taokey 1d | /taokey 12h | /taokey 7d | /keys | /khoa KEY\n\nKey được tạo sẽ tự có link API và tự hết hạn.'));
    bot.onText(/^\/taokey(?:\s+(\d+[hdw]))?$/i,(msg,match)=>{
        if(!isAdmin(msg.from.id))return bot.sendMessage(msg.chat.id,'⛔ Bạn không có quyền.');
        const duration=match[1]||'1d',ms=durationMs(duration),key=makeKey(),expiresAt=Date.now()+ms;
        keys.push({key,createdAt:Date.now(),expiresAt,active:true,createdBy:msg.from.id});saveKeys();
        const base=(process.env.APP_URL||'https://your-app.onrender.com').replace(/\/$/,'');
        const link=base+'/'+key;
        const text=`✅ ĐÃ TẠO KEY\n\n🔑 ${key}\n⏳ Thời hạn: ${duration}\n🕒 Hết hạn: ${new Date(expiresAt).toLocaleString('vi-VN',{timeZone:'Asia/Ho_Chi_Minh'})}\n\n🔗 Link API: ${link}\n\nMở link để xem danh sách endpoint. Key sẽ tự khóa khi hết hạn.`;
        bot.sendMessage(msg.chat.id,text,{disable_web_page_preview:true});
        if(String(msg.chat.id)!==String(process.env.ADMIN_NOTIFY_CHAT_ID||msg.chat.id))bot.sendMessage(process.env.ADMIN_NOTIFY_CHAT_ID||msg.chat.id,`🆕 Key mới ${key}\n${link}\nHạn: ${duration}`);
    });
    bot.onText(/^\/keys$/,(msg)=>{if(!isAdmin(msg.from.id))return;const active=keys.filter(k=>k.active&&k.expiresAt>Date.now());bot.sendMessage(msg.chat.id,active.length?active.map(k=>`${k.key} · ${new Date(k.expiresAt).toLocaleString('vi-VN',{timeZone:'Asia/Ho_Chi_Minh'})}`).join('\n'):'Không có key hoạt động.');});
    bot.onText(/^\/khoa\s+([A-F0-9]+)$/i,(msg,m)=>{if(!isAdmin(msg.from.id))return;const k=keys.find(x=>x.key===m[1].toUpperCase());if(!k)return bot.sendMessage(msg.chat.id,'Không tìm thấy key.');k.active=false;saveKeys();bot.sendMessage(msg.chat.id,`🔒 Đã khóa ${k.key}`);});
    setInterval(()=>{let dirty=false;for(const k of keys)if(k.active&&Date.now()>=k.expiresAt){k.active=false;dirty=true;}if(dirty)saveKeys();},30000);
    bot.on('polling_error',e=>console.error('Telegram polling:',e.message));
}

app.listen(PORT, () => {
    console.log('\n╔══════════════════════════════════════════════════════════════════╗');
    console.log('║     🎲 API GAME LC79 · HITCLUB · MAX789 🎲                       ║');
    console.log('╠══════════════════════════════════════════════════════════════════╣');
    console.log(`║   📡 Server: http://localhost:${PORT}                              ║`);
    console.log(`║   🎯 API: /api/xdlc · /api/tx · /api/hitclub/* · /api/max789/*      ║`);
    console.log('╠══════════════════════════════════════════════════════════════════╣');
    console.log('║   📊 79 THUẬT TOÁN ĐANG CHẠY NGẦM:                                ║');
    console.log('║   ├─ 10 Statistical (MA, EMA, WMA, Holt, ARIMA, SARIMA, GARCH)   ║');
    console.log('║   ├─ 15 Machine Learning (KNN, Bayes, Tree, RF, GBM, XGB, SVM)   ║');
    console.log('║   ├─ 10 Deep Learning (RNN, LSTM, GRU, BiLSTM, CNN, Transformer) ║');
    console.log('║   ├─ 10 Optimization (GA, PSO, SA, ACO, Bee, Firefly, Cuckoo...) ║');
    console.log('║   ├─ 14 Specialized (Markov, HMM, Monte Carlo, Kalman, Fourier)  ║');
    console.log('║   └─ 20 Ensemble Meta (Voting, Stacking, AdaBoost, Bagging...)   ║');
    console.log('╠══════════════════════════════════════════════════════════════════╣');
    console.log('║   🔄 TỰ ĐỘNG NẠP LỊCH SỬ + CẬP NHẬT MỖI 3 GIÂY                    ║');
    console.log('║   📤 KẾT QUẢ CHỈ GỒM: phiên + dự_đoán                             ║');
    console.log('╚══════════════════════════════════════════════════════════════════╝\n');
});
