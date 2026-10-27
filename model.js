/* Varanasi nowcast engine (runs in the browser or in node).
   rain probability = logistic regression over logits of [HistGradientBoosting, RandomForest, MLP]
   horizon          = min(30 min, time until the next IMD observation is due)                      */
var Nowcast=(function(){
var RAIN_MM=0.1,MAX_H=30;
function sig(z){return 1/(1+Math.exp(-z))}
function logit(p){p=Math.min(Math.max(p,1e-4),1-1e-4);return Math.log(p/(1-p))}
function walk(t,x){var n=0;while(t.f[n]>=0){n=x[t.f[n]]<=t.t[n]?t.l[n]:t.r[n]}return t.v[n]}
function gb(m,x){var s=m.base;for(var i=0;i<m.trees.length;i++)s+=walk(m.trees[i],x);return sig(s)}
function rf(trees,x){var s=0;for(var i=0;i<trees.length;i++)s+=walk(trees[i],x);return s/trees.length}
function nn(m,x){
  var z=x.map(function(v,i){return (v-m.mu[i])/m.sd[i]});
  for(var l=0;l<m.W.length;l++){
    var W=m.W[l],b=m.b[l],o=[];
    for(var j=0;j<b.length;j++){var s=b[j];for(var i=0;i<z.length;i++)s+=z[i]*W[i][j];o.push(l<m.W.length-1?Math.max(s,0):s)}
    z=o;
  }
  return sig(z[0]);
}
function parts(v,x){return [gb(v.gb,x),rf(v.rf,x),nn(v.nn,x)]}
function rainProb(v,x){
  var s=v.stack,p=parts(v,x);
  return sig(s.intercept+s.coef[0]*logit(p[0])+s.coef[1]*logit(p[1])+s.coef[2]*logit(p[2]));
}
function clip(r){return Math.min(Math.max(r,1),100)}
function dewdep(T,rh){var g=Math.log(rh/100)+17.62*T/(243.12+T);return T-243.12*g/(17.62-g)}
/* o: {T,RH,U,V:[now,-1h,-2h,-3h], P?:[..], R?:[mm last hour, -1h, -2h], ms, S:{elev,builtup,water,tree}} */
function feats(o,names){
  var rh=o.RH.map(clip),W=function(k){return Math.hypot(o.U[k],o.V[k])},d=new Date(o.ms);
  var hour=d.getUTCHours(),doy=Math.floor((o.ms-Date.UTC(d.getUTCFullYear(),0,0))/86400000);
  var f={temp_c:o.T[0],rh:rh[0],dewdep:dewdep(o.T[0],rh[0]),u:o.U[0],v:o.V[0],wspd:W(0),
    dT1:o.T[0]-o.T[1],dT3:o.T[0]-o.T[3],dRH1:rh[0]-rh[1],dRH3:rh[0]-rh[3],dws1:W(0)-W(1),
    hour_sin:Math.sin(2*Math.PI*hour/24),hour_cos:Math.cos(2*Math.PI*hour/24),
    doy_sin:Math.sin(2*Math.PI*doy/365.25),doy_cos:Math.cos(2*Math.PI*doy/365.25),
    elev:o.S.elev,builtup:o.S.builtup,water:o.S.water,tree:o.S.tree};
  if(o.P){f.dp1=o.P[0]-o.P[1];f.dp3=o.P[0]-o.P[3]}
  if(o.R){f.rain0=o.R[0];f.rain1=o.R[1];f.rain2=o.R[2];f.rain3sum=o.R[0]+o.R[1]+o.R[2]}
  return names.map(function(n){return f[n]});
}
function km(la1,lo1,la2,lo2){var dy=(la2-la1)*111.19,dx=(lo2-lo1)*111.19*Math.cos((la1+la2)/2*Math.PI/180);return Math.hypot(dx,dy)}
function nearestCell(grid,lat,lon){
  var b=null,bd=1e9;
  grid.cells.forEach(function(c){var d=km(lat,lon,c.lat,c.lon);if(d<bd){bd=d;b=c}});
  return {cell:b,km:bd};
}
/* inverse-distance weighting of per-station arrays (need = minimum array length) */
function idw(st,lat,lon,get,need){
  var num=null,den=0,n=0,ms=0;need=need||4;
  st.forEach(function(s){
    var a=get(s);
    if(!a||a.length<need||a.slice(0,need).some(function(v){return v==null||isNaN(v)}))return;
    var w=1/Math.pow(Math.max(km(lat,lon,s.lat,s.lon),0.5),2);
    if(!num)num=a.slice(0,need).map(function(){return 0});
    for(var i=0;i<need;i++)num[i]+=w*a[i];
    den+=w;n++;ms=Math.max(ms,Date.parse(s.obs_time));
  });
  return num?{v:num.map(function(x){return x/den}),n:n,ms:ms}:null;
}
function uv(s,which){
  if(!s.wind_ms||!s.wind_dir)return null;
  return s.wind_ms.map(function(w,i){var r=s.wind_dir[i]*Math.PI/180;return which=='u'?-w*Math.sin(r):-w*Math.cos(r)});
}
function r5(x){return Math.max(5,Math.round(x/5)*5)}
/* p = P(rain in the 30 min after the observation); hi/lo = decision thresholds from training */
function describe(p,rainNow,h,age,hi,lo){
  var none={headline:'No rain expected',detail:'Next '+h+' minutes'},m;
  if(rainNow===null){                      /* reduced model (no rain history): high false-alarm rate, so be cautious */
    if(p>=0.5)return {headline:'Rain likely',detail:'Within the next '+h+' minutes'};
    if(p>=hi)return {headline:'Rain possible',detail:'Within the next '+h+' minutes'};
    return none;
  }
  if(!rainNow){
    if(p>=hi){
      m=r5(MAX_H*(1-(p-hi)/(1-hi))-age);m=Math.min(m,Math.max(5,h));
      return {headline:'Rain expected in ~'+m+' min',detail:'Estimate, within the next '+h+' minutes',minutes:m};
    }
    return p>=lo?{headline:'Rain possible',detail:'Within the next '+h+' minutes'}:none;
  }
  if(p>=hi)return {headline:'Raining, likely to continue',detail:'Next '+h+' minutes'};
  if(p>=lo)return {headline:'Rain may ease soon',detail:'Uncertain, within the next '+h+' minutes'};
  m=Math.min(r5(MAX_H*(p/lo)-age),Math.max(5,h));
  return {headline:'Rain likely to stop in ~'+m+' min',detail:'Estimate',minutes:m};
}
function compass(d){return ['N','NE','E','SE','S','SW','W','NW'][Math.round(d/45)%8]}
function trendNotes(cur,out,h){
  var n=[],dT=out.temp-cur.temp,dR=out.rh-cur.rh,dW=(out.wind-cur.wind)*3.6;
  if(Math.abs(dT)>=0.5)n.push('Temperature '+(dT>0?'rising':'falling')+' about '+Math.abs(dT).toFixed(1)+'\u00B0C by +'+h+' min');
  if(Math.abs(dR)>=4)n.push('Humidity '+(dR>0?'rising':'falling')+' to ~'+Math.round(out.rh)+'%');
  if(Math.abs(dW)>=4)n.push('Wind '+(dW>0?'picking up':'easing')+' to ~'+Math.round(out.wind*3.6)+' km/h '+compass(out.dir));
  return n;
}
function forecast(models,live,grid,lat,lon,nowMs){
  var st=live.stations||[];
  var nc=nearestCell(grid,lat,lon),S=nc.cell;
  var T=idw(st,lat,lon,function(s){return s.temp_c}),RH=idw(st,lat,lon,function(s){return s.rh}),
      U=idw(st,lat,lon,function(s){return uv(s,'u')}),V=idw(st,lat,lon,function(s){return uv(s,'v')});
  if(!T||!RH||!U||!V)return {ok:false,headline:'No usable IMD data',detail:'Temperature, humidity and wind observations are needed'};
  var Pp=idw(st,lat,lon,function(s){return s.pres_hpa}),R=idw(st,lat,lon,function(s){return s.rain_mm},3);
  var cadence=live.cadence_min||60,obsMs=T.ms,age=(nowMs-obsMs)/60000,horizon=Math.min(MAX_H,Math.floor(cadence-age));
  var cur={temp:T.v[0],rh:RH.v[0],wind:Math.hypot(U.v[0],V.v[0]),dir:(Math.atan2(-U.v[0],-V.v[0])*180/Math.PI+360)%360};
  var res={ok:true,cur:cur,stations:T.n,obsMs:obsMs,validUntilMs:obsMs+cadence*60000,horizon:horizon,age:age,
           warmup:st.some(function(s){return s.n_obs<4})};
  if(age>cadence*3){res.headline='IMD data is out of date';res.detail='No forecast until the next update';return res}
  if(horizon<=0){res.headline='Waiting for the next IMD update';res.detail='The last forecast has expired';return res}
  var full=!!(Pp&&R&&models.full),v=full?models.full:models.lite;
  var o={T:T.v,RH:RH.v,P:full?Pp.v:null,U:U.v,V:V.v,R:full?R.v:null,ms:obsMs,S:S};
  var p=rainProb(v,feats(o,v.features));
  var hi=v.threshold,lo=hi*0.6;
  var rainNow=full?R.v[0]>=RAIN_MM:null;
  var d=describe(p,rainNow,horizon,Math.max(age,0),hi,lo);
  res.headline=d.headline;res.detail=d.detail;res.minutes=d.minutes;res.prob=p;res.rainNow=rainNow;res.variant=full?'full':'lite';
  var tf=feats({T:T.v,RH:RH.v,U:U.v,V:V.v,P:null,R:null,ms:obsMs,S:S},models.trend.features);
  var z=tf.map(function(x,i){return (x-models.trend.mu[i])/models.trend.sd[i]});
  var dl=function(k){var t=models.trend.targets[k],s=t.intercept;for(var i=0;i<z.length;i++)s+=t.coef[i]*z[i];return s*(Math.max(age,0)+horizon)/60};
  var u2=U.v[0]+dl('u'),v2=V.v[0]+dl('v');
  res.outlook={temp:cur.temp+dl('temp_c'),rh:Math.min(100,Math.max(0,cur.rh+dl('rh'))),wind:Math.hypot(u2,v2),dir:(Math.atan2(-u2,-v2)*180/Math.PI+360)%360};
  res.notes=trendNotes(cur,res.outlook,horizon);
  res.staleWindow=age>MAX_H;
  return res;
}
return {forecast:forecast,rainProb:rainProb,parts:parts,feats:feats,describe:describe,idw:idw,uv:uv,nearestCell:nearestCell,km:km};
})();
if(typeof module!=='undefined')module.exports=Nowcast;
