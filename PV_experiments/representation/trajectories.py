"""Replay verified physical evaluations through an observer, without changing core code."""
from __future__ import annotations
import importlib,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import METHODS,ORACLE,LABELS,COLORS,clock_grid,check_close,finite,from_day_moments,circular_stats,season
from .common import Figures,save_csv,write_json,atomic_json,read_json,sha,style,line,plt

FIELDS=('power_w','gross_kwh','movement_kwh','net_kwh','tilt_deg','azimuth_deg','action_tilt','action_azimuth','delta_tilt_deg','delta_azimuth_deg','poa_wm2','cell_temp_c')
WEATHER=('dni_wm2','ghi_wm2','dhi_wm2','ambient_temperature_c','wind_speed_m_s','surface_albedo','solar_zenith_deg','solar_azimuth_deg')
ORACLE_FIELDS=('power_w','gross_kwh','tilt_deg','azimuth_deg','poa_wm2')


def shortest(a,b):return (a-b+180)%360-180


def representative_days(days):
    """Outcome-blind calendar rule, one middle chronological day per season."""
    groups={s:[] for s in ('DJF','MAM','JJA','SON')}
    for d in sorted(days):groups[season(pd.Timestamp(d).month)].append(d)
    return [v[(len(v)-1)//2] for v in groups.values() if v]


class Recorder:
    """Delegating adapter for the existing run_one(), only observes reset/step."""
    def __init__(self,env,method):self.env=env;self.method=method;self.records=[];self.info=[]
    @property
    def unwrapped(self):return self.env.unwrapped
    @property
    def horizon(self):return self.env.horizon
    def reset(self,*args,**kw):
        self.records=[];self.info=[];obs=self.env.reset(*args,**kw);u=self.unwrapped
        self.pose=(float(u.tilt),float(u.azimuth));return obs
    def step(self,action):
        u=self.unwrapped; previous=(float(u.tilt),float(u.azimuth)); a=finite(action)
        if a.shape!=(2,) or (np.abs(a)>1.00001).any():raise ValueError('Invalid command')
        if self.method=='fixed_no_motion':check_close(a,[0,0],'Fixed baseline action')
        if self.method=='sun_slew_limited':
            sp=u._solar_position(u.current_time)
            expected=[0.,0.] if sp.zenith>=90 else [np.clip((np.clip(sp.zenith,0,90)-previous[0])/u.max_delta_tilt,-1,1),np.clip(shortest(sp.azimuth,previous[1])/u.max_delta_azimuth,-1,1)]
            check_close(a,expected,'Astronomical current-time command',atol=1e-6)
        obs,reward,done,info=self.env.step(action)
        dt=float(info['delta_tilt_deg']);da=float(info['delta_azimuth_deg']); tilt=float(info['tilt']);az=float(info['azimuth'])
        if not (0<=tilt<=90 and 0<=az<360):raise ValueError('Panel orientation outside valid domain')
        check_close([dt,da],[tilt-previous[0],shortest(az,previous[1])],'Executed movement')
        check_close([tilt,shortest(az,(previous[1]+a[1]*u.max_delta_azimuth)%360)],[np.clip(previous[0]+a[0]*u.max_delta_tilt,0,90),0.],'Command to physical pose',atol=1e-5)
        if abs(dt)>u.max_delta_tilt+1e-6 or abs(da)>u.max_delta_azimuth+1e-6:raise ValueError('Slew limits violated')
        seconds=abs(dt)/u.slew_rate_tilt_deg_s+abs(da)/u.slew_rate_azimuth_deg_s
        if seconds>u.interval_hours*3600+1e-7:raise ValueError('Sequential actuator time exceeds interval')
        cost=u.movement_penalty*u.actuator_power_w*seconds/3.6e6
        if u.movement_cost_mode!='geometry':raise ValueError('Nonphysical movement cost mode')
        check_close(info['movement_cost'],cost,'Actuator energy',atol=1e-10)
        check_close(info['energy_kwh'],info['power_dc_w']*u.interval_hours/1000,'Physical power integration',atol=1e-10)
        check_close(reward,info['energy_kwh']-cost,'Net reward',atol=1e-10)
        if self.method=='fixed_no_motion':check_close([tilt,az,dt,da,cost],[*self.pose,0,0,0],'Fixed pose and cost')
        self.records.append([info['power_dc_w'],info['energy_kwh'],cost,reward,tilt,az,a[0],a[1],dt,da,info['poa_global'],info['cell_temperature_c']]);self.info.append(dict(info))
        return obs,reward,done,info


def oracle_from_weather(u,infos):
    """Certified instantaneous maximum of isotropic POA and monotone PVWatts DC.

    Reuse actual physical backend with the exact perturbed albedo/weather.
    No env.step, no movement/reward/feasible trajectory for this reference.
    """
    from mbpo.env.baseline_contract import isotropic_max_poa_tilt,certify_pvwatts_monotonicity
    from mbpo.env.pv_power_backend import compute_pv_power
    rows=[]
    for info in infos:
        if info['albedo_source'] not in ('nsrdb_recorded','nsrdb_recorded_with_assumed_uncertainty'):raise ValueError('Oracle requires paired NSRDB albedo provenance')
        z,az,dn,gh,dh=map(float,[info['solar_zenith_deg'],info['solar_azimuth_deg'],info['dni_wm2'],info['ghi_wm2'],info['dhi_wm2']]);rho=float(info['surface_albedo'])
        tilt=isotropic_max_poa_tilt(z,dn,gh,dh,rho); az=az%360
        res=compute_pv_power(tilt,az,z,az,dn,gh,dh,area=u.area,efficiency=u.efficiency,power_model=u.power_model,temp_air_c=info['ambient_temperature_c'],wind_speed_ms=info['wind_speed_m_s'],wind_reference_height_m=u.wind_reference_height_m,gamma_pdc=u.gamma_pdc,sapm_mount=u.sapm_mount,wind_powerlaw_exponent=u.wind_powerlaw_exponent,surface_albedo=rho)
        if u.power_model!='pvwatts_sapm':raise ValueError('Expected PVWatts/SAPM model')
        certify_pvwatts_monotonicity(res,info['ambient_temperature_c'],u.gamma_pdc)
        rows.append([res.power_w,res.power_w*u.interval_hours/1000,tilt,az,res.poa_global_wm2])
    return finite(rows)


def load_physics_config(cfg,root):
    from MBPO_workspace.src.config import load_config
    from PV_experiments.core import uncertainty_path
    overrides={k:cfg[k] for k in ('model_epochs','model_max_train_samples') if cfg.get(k) is not None}
    return load_config(weather_uncertainty=read_json(uncertainty_path(cfg,root)),model_overrides=overrides)


def collect(cfg,split,data,stage,scope,root,out):
    from PV_experiments.core import run_path
    from SAC_workspace.src.environment.environment import SACEnvironment
    mc=importlib.import_module('MBPO_workspace.workflows.09_monte_carlo_evaluate')
    if tuple(mc.METHODS)!=METHODS:raise ValueError('Current evaluation registry differs from audited representation contract')
    phys=load_physics_config(cfg,root); days=data['days'] if scope=='full' else representative_days(data['days']);M=data['draws'];
    plan=dict(schema=1,stage=stage,scope=scope,days=days,draws=M,seeds=data['seeds'],fields=list(FIELDS),weather_fields=list(WEATHER),oracle_fields=list(ORACLE_FIELDS),methods=list(METHODS),sources=data['sources'],collector_sha256=sha(Path(__file__)))
    planfile=out/'collection_plan.json';progress=out/'collection_progress.json'
    if planfile.exists():
        if read_json(planfile)!=plan:raise ValueError('Existing collection inputs or code changed; select a new trajectory tag')
    else:
        if any(out.iterdir()):raise ValueError('Unowned trajectory output directory')
        atomic_json(planfile,plan)
    files=read_json(progress) if progress.exists() else {}
    expected={f'seed{s}/{d}.npz' for s in data['seeds'] for d in days}
    if not set(files).issubset(expected):raise ValueError('Unexpected files in collection progress')
    for name,record in files.items():
        if sha(out/name)!=record['sha256']:raise ValueError(f'Corrupted completed trajectory: {name}')
    # Reject a partially written/unregistered file after an interrupted rename.
    if {p.relative_to(out).as_posix() for p in out.rglob('*.npz')}!=set(files):raise ValueError('Unregistered trajectory file after interruption; inspect it before removing only that incomplete record')
    if (out/'trajectory_manifest.json').exists():
        final=read_json(out/'trajectory_manifest.json')
        if set(files)!=expected or final.get('files')!={k:v['sha256'] for k,v in files.items()}:raise ValueError('Completed collection manifest mismatch')
        print('Verified completed trajectory collection; no replay needed.');return
    for si,seed in enumerate(data['seeds']):
        agents={m:mc.load_policy(phys,m,run_path(cfg,m,seed,root)/'latest.pt',cfg['device'],expected_protocol_sha256=split['protocol_sha256'],expected_seed=seed,expected_epoch=cfg['epochs']) for m in ('sac','mbpo')}
        env=SACEnvironment(phys.sac.eval_env)
        try:
            if env.horizon!=195 or env.unwrapped.randomize_initial_orientation:raise ValueError('Unexpected evaluation horizon or random initial pose')
            for sid in days:
                name=f'seed{seed}/{sid}.npz'
                if name in files:print(f'SKIP verified collection seed {seed} {sid}',flush=True);continue
                di=data['days'].index(sid); raw=np.empty((M,len(METHODS),195,len(FIELDS)),dtype=np.float64);weather=np.empty((M,195,len(WEATHER))); oracle=np.empty((M,195,len(ORACLE_FIELDS)));hashes=[]
                for draw in range(M):
                    expected_sha=data['fingerprints'][(sid,draw)]; paired=None
                    for mi,method in enumerate(METHODS):
                        rec=Recorder(env,method);res=mc.run_one(rec,method,sid,draw,agents.get(method)); arr=finite(rec.records)
                        if arr.shape!=(195,len(FIELDS)):raise ValueError('Incomplete rich trajectory')
                        sig=(res['profile_sha256'],res['pose'],res['timestamps_utc']);
                        if sig[0]!=expected_sha or not pd.DatetimeIndex(pd.to_datetime(sig[2],utc=True)).equals(clock_grid(sid)):raise ValueError('Replay weather/UTC differs from original evaluation')
                        check_close(sig[1],[30,180],'Configured common initial pose')
                        w=finite([[i[k] for k in WEATHER] for i in rec.info]);
                        if mi==0:
                            paired=sig;weather[draw]=w;oracle[draw]=oracle_from_weather(env.unwrapped,rec.info);hashes.append(expected_sha)
                        elif sig!=paired or not np.array_equal(w,weather[draw]):raise ValueError('Paired physical realization mismatch')
                        check_close(arr[:,1:4].sum(axis=0),data['daily'][si,di,draw,mi],'Rich replay versus existing evaluated daily energy',atol=1e-7)
                        if (oracle[draw,:,0]+np.maximum(1e-5,1e-7*np.abs(arr[:,0]))<arr[:,0]).any():raise ValueError('Gross oracle below feasible DC generation')
                        raw[draw,mi]=arr
                check_close(raw[:,:,:,0].mean(axis=0),data['moments'][si,di,:,:,0],'Rich replay mean power versus original',atol=1e-5)
                check_close(raw[:,:,:,0].std(axis=0,ddof=1),data['moments'][si,di,:,:,1],'Rich replay power SD versus original',atol=1e-5)
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
                fd,tmp=tempfile.mkstemp(prefix='.trajectory-',dir=p.parent)
                try:
                    with os.fdopen(fd,'wb') as stream:
                        np.savez_compressed(stream,raw=raw,weather=weather,oracle=oracle,profile_sha256=np.asarray(hashes,dtype='U64'),timestamps=np.asarray([t.isoformat() for t in clock_grid(sid)],dtype='U40'));stream.flush();os.fsync(stream.fileno())
                    if p.exists():raise FileExistsError(p)
                    os.replace(tmp,p)
                finally:
                    if os.path.exists(tmp):os.unlink(tmp)
                u=env.unwrapped
                audit=dict(min_oracle_margin_w=float((oracle[:,None,:,0]-raw[:,:,:,0]).min()),max_actuator_seconds=float((np.abs(raw[:,:,:,8])/u.slew_rate_tilt_deg_s+np.abs(raw[:,:,:,9])/u.slew_rate_azimuth_deg_s).max()),max_fixed_movement_kwh=float(raw[:,0,:,2].max()),checked_feasible_steps=int(M*len(METHODS)*195),actuator_parameters=dict(power_w=float(u.actuator_power_w),tilt_rate_deg_s=float(u.slew_rate_tilt_deg_s),azimuth_rate_deg_s=float(u.slew_rate_azimuth_deg_s),cost_gain=float(u.movement_penalty),max_delta_tilt_deg=float(u.max_delta_tilt),max_delta_azimuth_deg=float(u.max_delta_azimuth)))
                files[name]=dict(sha256=sha(p),audit=audit);atomic_json(progress,files);print(f'Collected {stage} seed {seed} {sid} ({M} paired draws)',flush=True)
        finally:env.close()
    if set(files)!=expected:raise ValueError('Incomplete rich trajectory collection')
    audits=[v['audit'] for v in files.values()]
    if any(a['actuator_parameters']!=audits[0]['actuator_parameters'] for a in audits):raise ValueError('Actuator parameters changed across collection')
    atomic_json(out/'baseline_audit.json',dict(scope=scope,stage=stage,days=days,seeds=data['seeds'],draws=M,checked_feasible_steps=sum(a['checked_feasible_steps'] for a in audits),min_oracle_margin_w=min(a['min_oracle_margin_w'] for a in audits),max_actuator_seconds=max(a['max_actuator_seconds'] for a in audits),max_fixed_movement_kwh=max(a['max_fixed_movement_kwh'] for a in audits),fixed_pose_deg=[30,180],actuator_parameters=audits[0]['actuator_parameters'],oracle='isotropic POA maximizer, DC monotonicity certified for each exact perturbed weather row; no movement or net energy',astronomical='current decision-time sun position, same slew bounds, darkness holds; no future-weather access',cost='executed angular movement, sequential motors; excludes overnight reset, holding/idle and other losses'))
    atomic_json(out/'trajectory_manifest.json',dict(**plan,files={k:v['sha256'] for k,v in files.items()},note='Full paired physical replay on selected days; oracle is gross only and unconstrained. No simulated net oracle.'))


def load_rich_day(directory,manifest,data,sid):
    arrays=[];weather=None;oracle=None
    for si,seed in enumerate(data['seeds']):
        name=f'seed{seed}/{sid}.npz';p=directory/name
        if sha(p)!=manifest['files'][name]:raise ValueError('Rich trajectory content changed')
        with np.load(p,allow_pickle=False) as z:
            raw=finite(z['raw']);w=finite(z['weather']);o=finite(z['oracle']);ids=list(z['profile_sha256']);time=list(z['timestamps'])
        M=data['draws'];di=data['days'].index(sid)
        if raw.shape!=(M,4,195,len(FIELDS)) or w.shape!=(M,195,len(WEATHER)) or o.shape!=(M,195,len(ORACLE_FIELDS)):raise ValueError('Rich trajectory shape mismatch')
        if ids!=[data['fingerprints'][(sid,r)] for r in range(M)] or time!=[t.isoformat() for t in clock_grid(sid)]:raise ValueError('Rich trajectory provenance mismatch')
        check_close(raw[:,:,:,1:4].sum(axis=2),data['daily'][si,di],'Stored rich replay energy totals');check_close(raw[:,:,:,1]-raw[:,:,:,2],raw[:,:,:,3],'Rich net accounting')
        check_close(raw[:,:,:,0].mean(axis=0),data['moments'][si,di,:,:,0],'Stored rich power mean',atol=1e-5)
        if (o[:,None,:,0]+np.maximum(1e-5,1e-7*np.abs(raw[:,:,:,0]))<raw[:,:,:,0]).any():raise ValueError('Oracle below feasible generation')
        if weather is None:weather=w;oracle=o
        else:
            check_close(w,weather,'Weather across seeds');check_close(o,oracle,'Oracle across seeds');check_close(raw[:,:2],arrays[0][:,:2],'Physical baseline across seeds',atol=1e-7)
        arrays.append(raw)
    return np.stack(arrays),weather,oracle


def render_rich(directory,data,stage,scope,out):
    directory=Path(directory);m=read_json(directory/'trajectory_manifest.json');days=data['days'] if scope=='full' else representative_days(data['days']);M=data['draws'];S=len(data['seeds']);
    required={'schema':1,'stage':stage,'scope':scope,'days':days,'draws':M,'seeds':data['seeds'],'fields':list(FIELDS),'weather_fields':list(WEATHER),'oracle_fields':list(ORACLE_FIELDS),'methods':list(METHODS),'sources':data['sources']}
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Rich collection is stale or not the requested scope')
    if set(m['files'])!={f'seed{s}/{d}.npz' for s in data['seeds'] for d in days}:raise ValueError('Rich collection coverage mismatch')
    figs=Figures(out); moments={}; effort=[]; occupancy={method:np.zeros((18,36),dtype=float) for method in METHODS}; circ=[]; gap_heat=[]; selected=set(representative_days(days)); clock=8.75+np.arange(1,196)/12
    for sid in days:
        raw,w,oracle=load_rich_day(directory,m,data,sid) # seed × draw × method × time × field
        gap_heat.append((raw[:,:,METHODS.index('mbpo'),:,0]-raw[:,:,METHODS.index('sac'),:,0]).mean(axis=(0,1)))
        # Actual path integrals calculated BEFORE moments, retaining temporal covariance.
        for mi,method in enumerate(METHODS):
            p=raw[:,:,mi];vals={'power_w':p[:,:, :,0],'cumulative_gross_kwh':np.cumsum(p[:,:,:,1],axis=-1),'cumulative_net_kwh':np.cumsum(p[:,:,:,3],axis=-1),'cumulative_actuator_kwh':np.cumsum(p[:,:,:,2],axis=-1),'tilt_deg':p[:,:,:,4],'action_tilt':p[:,:,:,6],'action_azimuth':p[:,:,:,7]}
            for key,a in vals.items():moments.setdefault((method,key),[]).append((a.mean(axis=1),a.std(axis=1,ddof=1)))
            az,r=circular_stats(p[:,:,:,5],axis=(0,1));circ.extend(dict(scenario_id=sid,method=method,step=i+1,circular_mean_azimuth_deg=v,resultant_length=r[i]) for i,v in enumerate(az))
            hist,_,_=np.histogram2d(p[:,:,:,4].ravel(),p[:,:,:,5].ravel(),bins=(np.linspace(0,90,19),np.linspace(0,360,37)));occupancy[method]+=hist
            for si,seed in enumerate(data['seeds']):
                for draw in range(M):effort.append(dict(seed=seed,scenario_id=sid,draw=draw,method=method,tilt_travel_deg=np.abs(p[si,draw,:,8]).sum(),azimuth_travel_deg=np.abs(p[si,draw,:,9]).sum(),movement_kwh=p[si,draw,:,2].sum(),gross_kwh=p[si,draw,:,1].sum(),net_kwh=p[si,draw,:,3].sum(),diffuse_fraction=w[draw,:,2].sum()/w[draw,:,1].sum() if w[draw,:,1].sum()>0 else np.nan,oracle_gross_kwh=oracle[draw,:,1].sum()))
        for name,a in [('power_w',oracle[:,:,0]),('cumulative_gross_kwh',np.cumsum(oracle[:,:,1],axis=-1))]:
            a=np.broadcast_to(a,(S,)+a.shape);moments.setdefault((ORACLE,name),[]).append((a.mean(axis=1),a.std(axis=1,ddof=1)))
        for a,b in [('mbpo','sac'),('mbpo','sun_slew_limited')]:
            x=raw[:,:,METHODS.index(a)];y=raw[:,:,METHODS.index(b)]
            for key,v in [('power_difference_w',x[:,:,:,0]-y[:,:,:,0]),('cumulative_net_difference_kwh',np.cumsum(x[:,:,:,3]-y[:,:,:,3],axis=-1))]:moments.setdefault((f'{a}_minus_{b}',key),[]).append((v.mean(axis=1),v.std(axis=1,ddof=1)))
        for method in ('sac','mbpo','sun_slew_limited','fixed_no_motion'):
            gap=oracle[None,:,:,0]-raw[:,:,METHODS.index(method),:,0];moments.setdefault((method,'oracle_gross_power_gap_w'),[]).append((gap.mean(axis=1),gap.std(axis=1,ddof=1)))
        if sid in selected:
            # Deterministic, outcome-blind selected seed/draw, no averaged circular coordinates.
            fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
            for mi,method in enumerate(METHODS):
                p=raw[0,0,mi];line(axes[0],clock,p[:,0],method);line(axes[1],clock,p[:,4],method);axes[2].scatter(clock,p[:,5],s=6,color=COLORS[method],label=LABELS[method])
            line(axes[0],clock,oracle[0,:,0],ORACLE);axes[1].plot(clock,w[0,:,6],color='k',ls=':',label='Solar zenith');axes[2].scatter(clock,w[0,:,7],s=5,color='black',label='Sun azimuth');axes[0].set_ylabel('DC power (W)');axes[1].set_ylabel('Tilt (deg)');axes[2].set(ylabel='Azimuth (deg)',xlabel='UTC hour (unwrapped)',ylim=(0,360));axes[0].legend(ncol=2,fontsize=7);axes[1].legend(fontsize=7);axes[2].legend(ncol=2,fontsize=7)
            for ax in axes:style(ax)
            figs.save(fig,f'R01_daily_trajectory_{sid}','How does the policy move under the same realized weather?','Actual physical path, first selected training seed, draw 0; outcome-blind seasonal selection. Azimuth shown as points to avoid wrap artifacts.',f'{sid}.npz')
            fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
            for mi,method in enumerate(METHODS):
                p=raw[:,:,mi]
                for ax,field in zip(axes,(6,7)):
                    a=p[:,:,:,field].reshape(S*M,195);line(ax,clock,a.mean(axis=0),method,a.std(axis=0,ddof=0))
            axes[0].set_ylabel('Normalized tilt command');axes[1].set(ylabel='Normalized azimuth command',xlabel='UTC hour (unwrapped)');axes[0].legend(ncol=2,fontsize=8)
            for ax in axes:style(ax)
            figs.save(fig,f'R02_daily_actions_{sid}','How much control effort is commanded during the day?','Same historical day; equal seed/draw empirical mean ± population SD. Commands are increments, not angles.',f'{sid}.npz')
    # Compact sufficient statistics over all collected historical days.
    summary={}; rows=[]
    for (method,key),pairs in moments.items():
        mu=np.stack([p[0] for p in pairs],axis=1); sd=np.stack([p[1] for p in pairs],axis=1);s=from_day_moments(mu,sd,M);summary[(method,key)]=s;rows.extend(dict(method=method,quantity=key,step=i+1,utc_hour_unwrapped=clock[i],n_days=len(days),**{k:v[i] for k,v in s.items()}) for i in range(195))
    save_csv(out,'rich_curve_statistics.csv',rows);save_csv(out,'circular_azimuth_statistics.csv',circ);eff=pd.DataFrame(effort);save_csv(out,'daily_effort_and_oracle.csv',eff)
    for names,keys,filename,question in [((METHODS+(ORACLE,)),('cumulative_gross_kwh','cumulative_net_kwh','cumulative_actuator_kwh'),'R03_cumulative_energy','Where during the day do gross gains and movement costs accumulate?'),(('mbpo_minus_sac','mbpo_minus_sun_slew_limited'),('power_difference_w','cumulative_net_difference_kwh'),'R04_paired_response','When does MBPO gain or lose relative to competing feasible controllers?'),(METHODS,('oracle_gross_power_gap_w',),'R05_oracle_gap','How far are feasible controllers from instantaneous gross generation capacity?')]:
        fig,axes=plt.subplots(len(keys),1,figsize=(9,3.3*len(keys)),sharex=True,squeeze=False)
        for ax,key in zip(axes[:,0],keys):
            for name in names:
                if (name,key) not in summary:continue
                z=summary[(name,key)];line(ax,clock,z['mean'],name,z['sd_total'])
            ax.set_ylabel(key.replace('_',' '));ax.legend(fontsize=8,ncol=2);style(ax)
            if 'difference' in key:ax.axhline(0,color='black',lw=.7)
        axes[-1,0].set_xlabel('UTC hour (unwrapped)');figset_title=f'{scope.capitalize()} scope: {len(days)} historical days';fig.suptitle(figset_title)
        figs.save(fig,filename,question,'Mean ± empirical mixture SD across seed/day/weather; pairing and path integration precede aggregation. No oracle net/cost.', 'rich_curve_statistics.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,method in zip(axes.flat,METHODS):
        h=occupancy[method];h=h/h.sum();im=ax.imshow(h,origin='lower',aspect='auto',extent=[0,360,0,90],vmin=0,vmax=max((a/a.sum()).max() for a in occupancy.values()),cmap='viridis');ax.set_title(LABELS[method]);ax.set(xlabel='Azimuth (deg)',ylabel='Tilt (deg)');save_csv(out,f'occupancy_{method}.csv',pd.DataFrame(h))
    fig.subplots_adjust(right=.87,wspace=.25,hspace=.3);cax=fig.add_axes([.89,.17,.025,.66]);fig.colorbar(im,cax=cax,label='Fraction of recorded control intervals');fig._pv_layout_done=True;figs.save(fig,'R06_orientation_occupancy','Which physical orientations are used?','Equal 5-min interval occupancy across collected paired paths; includes dark intervals; azimuth is circular.', 'stored rich trajectories')
    # Paired gains by actual perturbed diffuse fraction, with each day/draw matched across methods.
    p=eff.pivot(index=['seed','scenario_id','draw','diffuse_fraction'],columns='method',values='net_kwh').reset_index();p['regime']=pd.cut(p.diffuse_fraction,[-np.inf,.35,.7,np.inf],labels=['<0.35','0.35–<0.70','≥0.70'],right=False);p['gain']=p.mbpo-p.sac
    reg=p.groupby(['seed','regime'],observed=True).gain.agg(['mean','count']).reset_index();save_csv(out,'diffuse_regime_paired_gain.csv',reg)
    fig,ax=plt.subplots(figsize=(7.5,4.4));
    for seed,g in reg.groupby('seed'):ax.plot(g.regime.astype(str),g['mean'],marker='o',label=f'Seed {seed}')
    ax.axhline(0,color='k',lw=.8);ax.set(xlabel='Integrated DHI / GHI regime (assumed weather draw)',ylabel='MBPO − SAC net energy (kWh/day)');ax.legend();style(ax);figs.save(fig,'R07_diffuse_regime_gain','Does paired benefit change with diffuse fraction?','Per-seed means of matched day/draw differences in predeclared diffuse-fraction bins; no inference over independent draws as historical days.','daily_effort_and_oracle.csv')
    # Daily integrated angular travel by calendar month (not cumulative orientation change).
    eff['month']=pd.to_datetime(eff.scenario_id).dt.month
    travel=eff.groupby(['seed','month','method'])[['tilt_travel_deg','azimuth_travel_deg']].mean().reset_index();save_csv(out,'monthly_travel_by_seed.csv',travel)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for ax,field in zip(axes,('tilt_travel_deg','azimuth_travel_deg')):
        for method,g in travel.groupby('method'):
            t=g.groupby('month')[field].agg(['mean','std']);line(ax,t.index.to_numpy(),t['mean'].to_numpy(),method,t['std'].to_numpy() if S>1 else None)
        ax.set_ylabel(field.replace('_',' '));style(ax)
    axes[0].legend(ncol=2,fontsize=7);axes[1].set_xlabel('Calendar month (collected days only)');axes[1].set_xticks(sorted(travel.month.unique()));figs.save(fig,'R08_angular_travel','How much actual angular movement is used by each controller?','Mean total absolute applied angular increments per day; sample SD of seed means; azimuth uses shortest wrapped movement.','monthly_travel_by_seed.csv')
    matrix=np.asarray(gap_heat);order=np.argsort(days);matrix=matrix[order];ordered=[days[i] for i in order];bound=max(float(np.abs(matrix).max()),1e-10)
    fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(matrix,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-bound,vmax=bound,extent=[clock[0]-1/24,clock[-1]+1/24,-.5,len(days)-.5]);ticks=np.unique(np.linspace(0,len(days)-1,min(12,len(days)),dtype=int));ax.set(yticks=ticks,yticklabels=[ordered[i] for i in ticks],xlabel='UTC hour (unwrapped)',ylabel='Historical scenario');fig.colorbar(im,ax=ax,label='Paired MBPO − SAC power (W)');save_csv(out,'paired_power_day_time_heatmap.csv',pd.DataFrame(matrix,columns=[f'step_{i+1}' for i in range(195)]).assign(scenario_id=ordered));figs.save(fig,'R09_paired_power_heatmap','When and on which days does MBPO improve generation?','Mean of matched seed/draw power differences for each collected day and UTC interval. Symmetric color scale about zero.','paired_power_day_time_heatmap.csv')
    figs.finish();write_json(out/'rich_scope.json',dict(stage=stage,scope=scope,n_days=len(days),days=days,draws=M,seeds=data['seeds'],note='Representative scope is not full partition. Oracle is only instantaneous gross DC bound.'))
