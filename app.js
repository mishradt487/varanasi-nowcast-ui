var map,marker,models,live,grid,sel=null,stationLayer;
function $(id){return document.getElementById(id)}
function ist(ms){return new Date(ms).toLocaleTimeString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:false})}
function compass(d){return ['N','NE','E','SE','S','SW','W','NW'][Math.round(d/45)%8]}
function wind(w,d){return Math.round(w*3.6)+' km/h '+compass(d)}
function tick(){$('clock').innerText=ist(Date.now())+' IST'}
function foot(){
  if(!live||!live.stations||!live.stations.length){$('foot').innerText='Waiting for IMD observations.';return}
  var ms=Math.max.apply(null,live.stations.map(function(s){return Date.parse(s.obs_time)}));
  $('foot').innerText='Observations: India Meteorological Department (latest '+ist(ms)+' IST)'+(live.demo?' \u2014 DEMO DATA':'')+
    '. Forecast: ML ensemble (gradient boosting + random forest + neural net, blended by logistic regression) trained on ERA5 and IMERG. Not an official forecast.';
}
function drawStations(){
  if(!map||!live)return;
  if(stationLayer)stationLayer.remove();
  stationLayer=L.layerGroup().addTo(map);
  (live.stations||[]).forEach(function(s){
    L.circleMarker([s.lat,s.lon],{radius:4,color:'#666',weight:1,fillColor:'#999',fillOpacity:.8,interactive:true})
      .bindTooltip(s.name+' \u00B7 '+(s.temp_c?s.temp_c[0].toFixed(1)+'\u00B0C':'')+' \u00B7 '+ist(Date.parse(s.obs_time))+' IST').addTo(stationLayer);
  });
}
async function refresh(){
  try{live=await (await fetch('live.json?t='+Date.now(),{cache:'no-store'})).json()}catch(e){live=null}
  foot();drawStations();
  if(sel)show();
}
async function start(){
  try{
    models=await (await fetch('models.json')).json();
    grid=await (await fetch('grid_static.json')).json();
  }catch(e){$('idle').innerText='Could not load the model files.';return}
  initMap();
  await refresh();
  setInterval(refresh,120000);
  setInterval(function(){if(sel)show()},30000);
}
function inside(a,b){var c=grid.cover;return a<=c.n&&a>=c.s&&b<=c.e&&b>=c.w}
function select(a,b){
  sel={lat:a,lng:b};
  $('idle').hidden=true;$('out').hidden=false;
  $('place').innerText=a.toFixed(4)+'\u00B0 N  '+b.toFixed(4)+'\u00B0 E';
  if(marker)marker.setLatLng([a,b]).addTo(map);
  show();
}
function text(h,d){$('headline').innerText=h;$('detail').innerText=d}
function clear(){['t0','t1','h0','h1','w0','w1','pct','meta'].forEach(function(c){$(c).innerText=''});$('fill').style.width='0';$('notes').innerHTML=''}
function show(){
  var a=sel.lat,b=sel.lng;
  if(!inside(a,b)){clear();text('Outside the covered area','The model only covers the Varanasi region');return}
  if(!live||!models){clear();text('Waiting for data','');return}
  var r=Nowcast.forecast(models,live,grid,a,b,Date.now());
  text(r.headline,r.detail||'');
  $('t0').innerText=r.cur?r.cur.temp.toFixed(1)+'\u00B0C':'';
  $('h0').innerText=r.cur?Math.round(r.cur.rh)+'%':'';
  $('w0').innerText=r.cur?wind(r.cur.wind,r.cur.dir):'';
  var o=r.outlook;
  $('hz').innerText=o?'In '+r.horizon+' min':'Later';
  $('t1').innerText=o?o.temp.toFixed(1)+'\u00B0C':'';
  $('h1').innerText=o?Math.round(o.rh)+'%':'';
  $('w1').innerText=o?wind(o.wind,o.dir):'';
  var p=r.prob;
  $('pct').innerText=p==null?'':Math.round(p*100)+'%';
  $('fill').style.width=p==null?'0':Math.round(p*100)+'%';
  $('notes').innerHTML=(r.notes||[]).map(function(n){return '<li>'+n+'</li>'}).join('');
  var m=[];
  if(r.obsMs){m.push('Based on IMD data from '+ist(r.obsMs)+' IST ('+Math.round(r.age)+' min ago, '+r.stations+' station'+(r.stations>1?'s':'')+'). Valid until '+ist(Math.min(r.validUntilMs,r.obsMs+(r.age+r.horizon)*60000))+' IST.')}
  if(r.variant==='lite')m.push('Rain history not available: using the reduced model.');
  if(r.warmup)m.push('Trend inputs still warming up (fewer than 4 hourly observations).');
  if(r.staleWindow)m.push('Observation is older than 30 min: skill is lower until the next update.');
  $('meta').innerText=m.join(' ');
}
function initMap(){
  var c=grid.cover;
  map=L.map('map',{center:[25.3176,82.9739],zoom:11,zoomControl:true,attributionControl:true,minZoom:9});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{subdomains:'abcd',maxZoom:18,
    attribution:'&copy; OpenStreetMap contributors &copy; CARTO'}).addTo(map);
  L.rectangle([[c.s,c.w],[c.n,c.e]],{color:'#4a4a4a',weight:1,fill:false,interactive:false}).addTo(map);
  marker=L.circleMarker([25.3176,82.9739],{radius:6,color:'#000',weight:2,fillColor:'#fff',fillOpacity:1});
  map.on('click',function(e){select(e.latlng.lat,e.latlng.lng)});
}
tick();setInterval(tick,30000);
start();
