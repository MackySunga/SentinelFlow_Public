const slides=[...document.querySelectorAll('.slide')];
const nav=document.getElementById('nav');
const counter=document.getElementById('counter');
let current=0;
slides.forEach((s,i)=>{const b=document.createElement('button');b.textContent=`${String(i+1).padStart(2,'0')} ${s.dataset.title}`;b.onclick=()=>show(i);nav.appendChild(b);});
function show(i){current=(i+slides.length)%slides.length;slides.forEach((s,idx)=>s.classList.toggle('active',idx===current));[...nav.children].forEach((b,idx)=>b.classList.toggle('active',idx===current));counter.textContent=`${current+1} / ${slides.length}`;}
document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(current+1);if(e.key==='ArrowLeft')show(current-1);});show(0);
const canvas=document.getElementById('fftCanvas');const ctx=canvas?.getContext('2d');let t=0;
function drawFFT(){if(!ctx)return;const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='rgba(55,220,255,.14)';ctx.lineWidth=1;for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
ctx.font='700 15px Segoe UI';ctx.fillStyle='#8da8c4';ctx.fillText('Time-domain traffic signal',32,32);ctx.fillText('Frequency-domain signature',520,32);
ctx.strokeStyle='#37dcff';ctx.lineWidth=3;ctx.beginPath();for(let x=30;x<430;x++){let burst=Math.sin((x+t)/21)*30+Math.sin((x+t)/7)*14;let spike=((x+Math.floor(t))%92<14)?55:0;let y=175-burst-spike;if(x===30)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
for(let i=0;i<12;i++){let x=535+i*28;let amp=30+Math.abs(Math.sin(t/28+i*.7))*90/(i*.38+1);ctx.fillStyle=i<3?'rgba(99,255,180,.86)':'rgba(55,220,255,.72)';ctx.fillRect(x,255-amp,18,amp);}
t+=2;requestAnimationFrame(drawFFT);}drawFFT();
