"""Fresh on-policy Stepwise-Flow-GRPO / endpoint-gradient training.

Supports one GPU or torchrun. All ranks own equally many trajectories of each
prompt group; only LoRA gradients are averaged, once per fresh optimizer update.
ODE branches supply stopped rewards/directions, never additional training rows.
"""
import argparse
import collections
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict, set_peft_model_state_dict

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from rvge.endpoint_gradient import trajectory_directions, velocity_ascent_loss
from rvge.stepwise_traj_window import trajectory_window_directions
from rvge.fdfo_aligned_loss import fdfo_first_order_velocity_loss
from rvge.stepwise_batch import UpdateBatch, select_prompt_ids, checkpoint_training_totals
from rvge.stepwise_batch import select_prompt_ids_weighted, PromptResidualTable, select_prompt_ids_by_source
from rvge.stepwise_credit import (flow_coefficients, ddim_coefficients, uniform_value_nodes, stochastic_steps,
    trainable_transitions,
    joint_gain_advantages, gaussian_logprob, grpo_loss, gaussian_kl, sde_coefficients, sde_schedule_of,
    SDE_SCHEDULES)
from rvge.stepwise_normalization import (GainNormalizer, REWARD_NORMALIZATIONS, batch_reward_scale,
    endpoint_step_squares, group_gain_statistics, inverse_rms_time_weights, parse_reward_scale, step_gain_statistics)
from rvge.stepwise_rows import (build_rows, shard_rows, sample_rows, sample_rows_paper, sample_rows_ode,
    gathered_order, group_positions)
from rvge.stepwise_group_fit import pooled_group_directions, group_slices, sibling_directions, cloud_directions, anchor_directions, select_trained, group_nearest_success_directions
from rvge.stepwise_advantage_fm import endpoint_advantages, select_endpoints, advantage_fm_loss
from rvge.stepwise_runtime import TrainingRuntime, SOURCES
from rvge.stepwise_geneval import reward_summary
multi_summary = QUALITY_KEYS = compose_multi_rewards = compose_multi_states = None  # multi-reward module not part of this release

def now():return datetime.now().astimezone().isoformat()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def sync():torch.cuda.synchronize()
def digest(state):
    h=hashlib.sha256()
    for n,p in sorted(state.items()):h.update(n.encode());h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()
def distributed():return dist.is_available() and dist.is_initialized()
def gather(x):
    if not distributed():return x
    parts=[torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts,x);return torch.cat(parts)
def allmax(x,device):
    a=torch.tensor(x,device=device,dtype=torch.float64)
    if distributed():dist.all_reduce(a,op=dist.ReduceOp.MAX)
    return float(a)

def reference_time_weight(t,mode,cut=None,exponent=None):
    """Time weight of the endpoint method's penalty toward the frozen base velocity.

'late5' is the historical (1-t)**5: it vanishes at the high-noise steps, which is where
layout, background and colour are decided, so drift there goes unpenalised.
'uniform' weighs every step alike, as
DiffusionNFT's KL does.
"""
    if mode=='late5':return (1-t)**5
    if mode=='uniform':return 1.
    # 'early': only transitions that start above `cut`.
    if mode=='early':return 1. if t>cut else 0.
    # 'power': t**exponent, a smooth version of 'early' (exponent 4 on the 10-step grid: 1 .85 .70 .54 .39 .25 .13 .05 .01 0).
    if mode=='power':return t**exponent
    raise ValueError(f'unknown reference time weight {mode!r}')

def dtype_from_name(name):
    mapping={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}
    try:return mapping[name]
    except KeyError as e:raise ValueError(f'invalid dtype {name!r}') from e

def load_prompts(path):
    path=Path(path)
    if path.suffix=='.json':
        data=json.loads(path.read_text())
        if isinstance(data,dict):
            for key in ('train_prompts','eval_prompts','prompts'):
                if key in data:return list(data[key])
            raise ValueError(f'json prompt file {path} must contain a prompt list')
        if isinstance(data,list):return [str(p) for p in data]
        raise ValueError(f'json prompt file {path} must be a list or dict of prompts')
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

@torch.no_grad()
def grouped_endpoints(rt,prompt,firsts,batch,chunk):
    """Advance every stopped ODE suffix, batching the ones sharing a time node.

Suffix q leaves its main state at time index q+1, so at node k the suffixes
q<k are all on that node and go through the transformer together. The sample
count is unchanged; only the number of launches and the batch shape change.
"""
    live=list(firsts)
    for k in range(1,len(rt.times)-1):
        nodes=torch.cat(live[:k])
        advance=[]
        for piece in nodes.split(chunk or len(nodes)):
            advance.append(rt.velocity(piece,rt.times[k],prompt))
        advance=torch.cat(advance).split(batch)
        delta=rt.times[k+1]-rt.times[k]
        for i in range(k):live[i]=live[i]+delta*advance[i]
    return live


@torch.no_grad()
def sample_group(rt,prompt,args,step,group,rank,world):
    """Return only T main states/velocities, T+1 stopped endpoints/rewards."""
    batch=args.group_size//world
    schedule,noise_exponent=sde_schedule_of(args)
    key=args.seed*10**12+step*10**7+group*10**4
    x=rt.noise(12_000_000_000+key).expand(batch,-1,-1,-1).clone()
    xs=[];vs=[];children=[];ys=[];rs=[];bs=[];taus=[];mus=[]
    f0=rt.forward_samples;n0=rt.nfe;s0=rt.score_samples
    score_seconds=0.;realized=[]
    grouped=args.rollout_batching=='grouped' and args.method=='endpoint'
    firsts=[]
    for q,(t,s) in enumerate(zip(rt.times,rt.times[1:])):
        v=rt.velocity(x,t,prompt)
        xs.append(x.cpu());vs.append(v.cpu())
        if grouped:
            firsts.append(x+(rt.times[q+1]-t)*v)
        else:
            nodes=uniform_value_nodes(t,args.value_substeps) if args.method=='paper' else rt.times[q:]
            y=x+(nodes[1]-nodes[0])*v
            for u,w in zip(nodes[1:-1],nodes[2:]):y=y+(w-u)*rt.velocity(y,u,prompt)
            ys.append(y.cpu());sync();tic=time.monotonic()
            rs.append(rt.score(y,prompt).cpu());sync();score_seconds+=time.monotonic()-tic
        if args.method=='paper':
            a,b,tau=(flow_coefficients(t,s,args.paper_noise,rt.times[1]) if args.paper_sampler=='flow'
                     else ddim_coefficients(t,s,args.ddim_eta))
            a=torch.full((batch,1,1,1),a,device=rt.device)
            b=torch.full_like(a,b);tau=torch.full_like(a,tau)
        else:
            if args.sde_noise_mode=='state_rms':
                rms=(x+(s-t)*v).double().flatten(1).square().mean(1).sqrt().float()
                unit_tau=flow_coefficients(t,s,1.,rt.times[1])[2]
                level=(args.noise_radius*rms/unit_tau).view(-1,1,1,1)
            else:
                level=torch.full((batch,1,1,1),args.noise_radius,device=rt.device,dtype=torch.float32)
            a,b,tau=sde_coefficients(t,s,level,rt.times[1],schedule,noise_exponent)
        eps=torch.cat([rt.noise(21_000_000_000+key+q*97+(rank*batch+i)*1009) for i in range(batch)])
        mu=a*x+b*v
        child=mu+tau*eps
        realized.extend((tau*eps).double().flatten(1).square().mean(1).sqrt().cpu().tolist())
        children.append(child.cpu());mus.append(mu.cpu());bs.append(b.cpu());taus.append(tau.cpu())
        x=child
    if grouped:
        live=grouped_endpoints(rt,prompt,firsts,batch,args.rollout_chunk)
        sync();tic=time.monotonic()
        allscores=rt.score(torch.cat(live+[x]),prompt,args.score_chunk).cpu()
        sync();score_seconds+=time.monotonic()-tic
        ys=[y.cpu() for y in live]+[x.cpu()];rs=list(allscores.split(batch))
    else:
        ys.append(x.cpu());sync();tic=time.monotonic();rs.append(rt.score(x,prompt).cpu())
        sync();score_seconds+=time.monotonic()-tic
    T_=len(rt.times)-1
    expected=T_*args.value_substeps if args.method=='paper' else T_*(T_+1)//2
    assert rt.forward_samples-f0==expected*batch
    assert rt.score_samples-s0==(T_+1)*batch
    result=dict(x=torch.stack(xs,1),old=torch.stack(vs,1),child=torch.stack(children,1),
                mean=torch.stack(mus,1),b=torch.stack(bs,1),tau=torch.stack(taus,1),
                y=torch.stack(ys,1),r=torch.stack(rs,1))
    return result,dict(forward_calls=rt.nfe-n0,forward_samples=rt.forward_samples-f0,
        score_samples=rt.score_samples-s0,score_seconds=score_seconds,
        innovation_rms_mean=float(np.mean(realized)))

def train(args):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    update_batch=UpdateBatch(args.group_size,args.groups_per_update,world,main_steps=args.main_steps,batching=args.batching)
    if args.expected_trajectories_per_update is not None and args.expected_trajectories_per_update!=update_batch.trajectories_per_update:
        raise ValueError('group size times groups per update does not match expected trajectories per update')
    if args.train_microbatch<=0:raise ValueError('train microbatch must be positive')
    if args.actor_loss=='fdfo_first_order' and (args.method!='endpoint' or args.direction_normalization!='none'):
        raise ValueError('FDFO first-order loss requires endpoint estimates with final normalization disabled')
    if args.batching=='rows':
        # Row-major is now implemented for both estimators. It changes only which rank
        # holds which trajectory and the batch composition of each transformer call -
        # the seeds, and therefore the sampled set, are unchanged. For --method paper
        # it is what makes the wall-clock axis an algorithm comparison rather than a
        # batching one: prompt-major leaves group_size/world rows per launch.
        if args.train_microbatch>update_batch.local_trajectories_per_update:
            raise ValueError('train microbatch exceeds this rank\'s trajectory shard')
        if args.shard_mode=='block' and update_batch.local_trajectories_per_update%args.group_size:
            raise ValueError('block sharding needs a per-rank row count that is a multiple of the group size')
    if args.fit_scope in('pool_head','nearest_success_group') and not (args.batching=='rows' and args.shard_mode=='block'):
        raise ValueError('--fit-scope pool_head needs --batching rows --shard-mode block (whole groups per rank)')
    if args.state_select!='all' and not (args.batching=='rows' and args.method=='endpoint' and args.rollout_mode=='sde'):
        raise ValueError('--state-select needs --batching rows --method endpoint on the SDE rollout')
    if args.fm_weight and not (args.batching=='rows' and args.shard_mode=='block' and args.method=='endpoint'):
        raise ValueError('--fm-weight needs --method endpoint --batching rows --shard-mode block')
    if args.rollout_mode=='ode_group':
        if not (args.batching=='rows' and args.shard_mode=='block' and args.method=='endpoint' and args.independent_roots):
            raise ValueError('--rollout-mode ode_group needs --method endpoint --batching rows --shard-mode block --independent-roots')
        if args.direction_index=='next' or args.reward_normalization!='none':
            raise ValueError('--rollout-mode ode_group fits at the current state on raw rewards (--reward-normalization none)')
        if args.train_per_group is not None and not 0<args.train_per_group<=args.group_size:
            raise ValueError('--train-per-group must be between 1 and --group-size')
        if args.mixed_extra_k and (args.cloud_size or args.direction_mode!='sibling' or args.fm_weight):
            raise ValueError('--mixed-extra-k is implemented for the plain sibling ode_group path only')
        if args.cloud_size and args.group_size%(1+args.cloud_size):
            raise ValueError('--group-size must be a multiple of 1 + --cloud-size (M mains x (1 + C))')
        if args.cloud_size and not 0<args.cloud_strength<1:
            raise ValueError('--cloud-strength must be in (0, 1)')
    if args.fit_scope=='pool_head' and args.direction_index=='next':
        raise ValueError('--fit-scope pool_head fits the direction at the current state only')
    local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=90))
    device=f'cuda:{local}';out=args.output.resolve();started=time.monotonic()
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if distributed():dist.barrier()
    def status(phase,**extra):
        write(out/f'rank{rank:02d}_status.json',dict(status=phase,pid=os.getpid(),rank=rank,world=world,
            updated_at=now(),elapsed_seconds=time.monotonic()-started,**extra))
        if rank==0:write(out/'status.json',dict(status=phase,coordinator_pid=os.getpid(),world_size=world,
            updated_at=now(),elapsed_seconds=time.monotonic()-started,**extra))
    try:
        status('loading_model')
        # str() also covers non-finite floats: --paper-max-btau defaults to inf, which
        # allow_nan=False refuses.
        cfg={k:str(v) if isinstance(v,Path) or (isinstance(v,float) and not np.isfinite(v)) else v
             for k,v in vars(args).items()}
        cfg.update(world_size=world,actor_dtype=args.runtime_dtype,scorer_dtype=args.scorer_dtype,
            local_trajectories=update_batch.local_trajectories_per_update,
            sde_noise_mode=args.sde_noise_mode,sde_schedule=args.sde_schedule,
            fit_head_rows=args.fit_head_rows,independent_roots=args.independent_roots,
            ema_update_interval=args.ema_update_interval,ema_warmup=args.ema_warmup,
            sde_noise_exponent=args.sde_noise_exponent,allow_tf32=False,resolution=512,cfg=1.,
            eval_cfg=args.eval_cfg,
            auxiliary_training_rows=0,main_steps=args.main_steps,trajectories_per_update=update_batch.trajectories_per_update,
            main_rows_per_update=update_batch.main_rows_per_update,
            local_trajectories_per_update=update_batch.trajectories_per_update//world,
            local_rows_per_update=update_batch.local_rows_per_update,
            reward={'pickscore':'raw PickScore float images',
                    'ocr':'PaddleOCR 1-edit_distance/len, the Flow-GRPO OCR reward on uint8 images',
                    'geneval':'GenEval partial score, reward_server/gen_eval.py (Mask2Former Swin-S + CLIP ViT-L/14) on uint8 images',
                    'multi':f'NFT Table 1 joint reward: q = PickScore/26 + ClipScore + HPSv2.1 (fp32 float images); pickapic {args.pickapic_scale}*q; geneval/ocr task + {args.quality_lambda}*{args.quality_mode}(q - ref {args.quality_ref})'}[args.reward],
            paper_logprob_reduction='mean over latent dimensions, inherited Flow-GRPO convention')
        if rank==0:
            write(out/'resolved_config.json',cfg)
            files=[Path(__file__),REPO/'rvge/endpoint_gradient.py',REPO/'rvge/stepwise_credit.py',REPO/'rvge/stepwise_runtime.py',REPO/'rvge/stepwise_batch.py',REPO/'rvge/fdfo_aligned_loss.py',REPO/'rvge/stepwise_rows.py',REPO/'rvge/stepwise_traj_window.py',REPO/'rvge/stepwise_normalization.py',REPO/'rvge/stepwise_ocr.py',REPO/'rvge/stepwise_geneval.py']
            snap=out/'source_snapshot';snap.mkdir()
            for f in files:(snap/f.name).write_bytes(f.read_bytes())
            write(out/'source_sha256.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
        data=json.loads(args.panel.read_text())
        prompts=data['train_prompts'];evalprompts=data['eval_prompts']
        if args.eval_prompts_file is not None:evalprompts=load_prompts(args.eval_prompts_file)
        if args.eval_prompt_limit is not None and args.reward!='multi':evalprompts=evalprompts[:args.eval_prompt_limit]
        if not prompts or not evalprompts:raise ValueError('empty prompt panel')
        if args.reward=='geneval':
            meta=data.get('metadata') or {}
            missing=[p for p in prompts+evalprompts if p not in meta]
            if missing:raise ValueError(f'{len(missing)} prompts have no GenEval metadata, e.g. {missing[0]!r}')
        prompt_sources=None;train_sources=None
        if args.reward=='multi':
            # Multi panel (scripts/build_multi_panel.py): one source per train / eval row; GenEval rows carry metadata.
            train_sources=list(data['train_sources']);eval_sources=list(data['eval_sources'])
            if args.eval_prompts_file is not None:
                raise ValueError('the multi reward evaluates the panel eval rows (sources attached); no prompt-file override')
            if len(train_sources)!=len(prompts) or len(eval_sources)!=len(evalprompts):raise ValueError('multi panel sources misaligned')
            if args.eval_prompt_limit is not None:
                # Per-source cap: the first N eval rows of every source (smoke tests, cheaper in-training readouts).
                keep=[];seen=collections.Counter()
                for i_,s_ in enumerate(eval_sources):
                    if seen[s_]<args.eval_prompt_limit:keep.append(i_);seen[s_]+=1
                evalprompts=[evalprompts[i_] for i_ in keep];eval_sources=[eval_sources[i_] for i_ in keep]
            prompt_sources={}
            for p_,s_ in list(zip(prompts,train_sources))+list(zip(evalprompts,eval_sources)):
                if prompt_sources.setdefault(p_,s_)!=s_:raise ValueError(f'prompt in two sources: {p_!r}')
            meta=data.get('metadata') or {}
            missing=[p_ for p_,s_ in prompt_sources.items() if s_=='geneval' and p_ not in meta]
            if missing:raise ValueError(f'{len(missing)} GenEval prompts have no metadata, e.g. {missing[0]!r}')
            if args.source_shares is None:raise ValueError('--reward multi needs --source-shares')
            source_shares={k:float(v) for k,v in (kv.split('=') for kv in args.source_shares.split(','))}
            bad=sorted(set(source_shares)-set(SOURCES))
            if bad:raise ValueError(f'unknown sources in --source-shares: {bad}')
            quality_ref={k:float(v) for k,v in (kv.split('=') for kv in args.quality_ref.split(','))} if args.quality_ref else {}
            # --quality-margin: scalar (both rule sources) or per-source 'ocr=0.03,geneval=0'.
            if '=' in args.quality_margin:
                quality_margin={k:float(v) for k,v in (kv.split('=') for kv in args.quality_margin.split(','))}
                bad=sorted(set(quality_margin)-{'geneval','ocr'})
                if bad:raise ValueError(f'--quality-margin only takes the rule sources geneval/ocr, got {bad}')
            else:quality_margin=float(args.quality_margin)
        if args.unique_prompts_per_update:
            if len(set(prompts))!=len(prompts):raise ValueError('unique prompt sampling requires a panel with distinct prompt texts')
            if args.groups_per_update>len(prompts):raise ValueError('unique prompt groups exceed the available prompt panel')
        if rank==0:
            panel=dict(data);panel['eval_prompts']=evalprompts
            if args.eval_prompts_file is not None:panel['eval_source']=str(args.eval_prompts_file)
            if args.eval_prompt_limit is not None:panel['eval_prompt_limit']=args.eval_prompt_limit
            write(out/'panel.json',panel)
        rt=TrainingRuntime(args.seed,args.lora_rank,args.lora_alpha,device,steps=args.main_steps,
            runtime_dtype=dtype_from_name(args.runtime_dtype),scorer_dtype=dtype_from_name(args.scorer_dtype),
            trainable_dtype=None if args.trainable_dtype=='inherit' else dtype_from_name(args.trainable_dtype),
            score_chunk=args.score_chunk,gradient_checkpointing=not args.no_gradient_checkpointing,
            reward=args.reward,ocr_workers=args.ocr_workers,ocr_backend=args.ocr_backend,
            ocr_gpu_mem=args.ocr_gpu_mem,reward_metadata=data.get('metadata'),geneval_train_reward=args.geneval_train_reward,
            geneval_batch=args.geneval_batch,
            reward_vae_dtype=None if args.reward_vae_dtype=='inherit' else dtype_from_name(args.reward_vae_dtype),
            mix_pickscore_lambda=args.mix_pickscore_lambda,mix_pickscore_ref=args.mix_pickscore_ref,
            mix_pickscore_clamp=args.mix_pickscore_clamp,
            prompt_sources=prompt_sources,quality_lambda=args.quality_lambda,quality_mode=args.quality_mode,
            quality_ref=(quality_ref if args.reward=='multi' else None),pickapic_scale=args.pickapic_scale,
            quality_weights=({k:float(v) for k,v in (kv.split('=') for kv in args.quality_weights.split(','))} if args.quality_weights else None),
            quality_swap=not args.no_quality_swap)
        base_weights=torch.tensor(rt.times[:-1]).pow(args.time_exponent)
        if args.time_mode=='dt':base_weights*=torch.tensor(rt.times[:-1])-torch.tensor(rt.times[1:])
        base_weights/=base_weights.mean()
        weights=base_weights
        normalizer=GainNormalizer(args.reward_normalization,args.gain_std_decay,args.gain_std_fixed,args.gain_std_fixed_start)
        if normalizer.needs_update_statistics and args.batching!='rows':
            raise ValueError(f'--reward-normalization {args.reward_normalization} needs the whole update: use --batching rows')
        reward_scale=parse_reward_scale(args.reward_scale)
        if reward_scale is not None and args.reward_normalization!='none':
            raise ValueError('--reward-scale replaces normalization: use it with --reward-normalization none')
        if reward_scale=='batch_std' and args.batching!='rows':
            raise ValueError('--reward-scale batch_std needs the whole update: use --batching rows')
        # Which transitions carry a policy-gradient term. Three independent rules,
        # all in stepwise_credit.trainable_transitions: tau>0 (no Gaussian density on a
        # deterministic edge), Flow-GRPO's own timestep_fraction truncation, and an
        # optional |b|/tau ceiling for steps that are numerically deterministic edges.
        trainable_steps=(trainable_transitions(rt.times,args.paper_sampler,args.paper_noise,
                                               args.ddim_eta,args.paper_timestep_fraction,
                                               args.paper_max_btau)
                         if args.method=='paper' else list(range(len(rt.times)-1)))
        # Flow-GRPO's KL divides by std_dev_t**2 while its log density divides by
        # (std_dev_t*sqrt(-dt))**2, so the term beta=0.01 was tuned against is the exact
        # Gaussian KL times dt. Reproduce that scaling rather than silently running a
        # regulariser 1/dt = 3.7x..25x stronger than anything anyone tuned. The DDIM
        # schedule needs no correction: there sigma_t is the transition std itself, so
        # the paper's Eq.(9) policy and the exact KL already agree.
        paper_kl_scale=[1.]*(len(rt.times)-1)
        if args.method=='paper' and args.paper_kl_convention=='flow_grpo' and args.paper_sampler=='flow':
            paper_kl_scale=[float(t)-float(s) for t,s in zip(rt.times,rt.times[1:])]
        if not trainable_steps:raise ValueError('sampler leaves no stochastic transition to train on')
        trainable_set=set(trainable_steps)
        # Rows above --policy-time-cut still run forward (and carry the reference penalty) so the row
        # accounting is unchanged; their policy term is a zero that keeps the graph for backward.
        policy_on=lambda t:((args.policy_time_cut is None or t<=args.policy_time_cut) and
                            (args.policy_time_min is None or t>=args.policy_time_min))
        unit_tau=[float(sde_coefficients(t,s,1.,rt.times[1],args.sde_schedule,args.sde_noise_exponent)[2])
                  for t,s in zip(rt.times,rt.times[1:])]
        # Per-transition weight of the reference penalty on mean((v-ref)**2), added to the reward loss.
        # 'flow_grpo' is Flow-GRPO's train_sd3.py term mean((mu-mu_ref)**2)/(2*std_dev_t**2) in velocity form:
        # mu-mu_ref = b*(v-ref) and std_dev_t**2 = tau**2/(t-s) on this run's own SDE (eta = --noise-radius),
        # so the weight is b**2*(t-s)/(2*tau**2); it grows toward the late steps like dt**2*(1-t)/t.
        def flow_grpo_weight(t,s):
            _,b,tau=sde_coefficients(t,s,args.noise_radius,rt.times[1],args.sde_schedule,args.sde_noise_exponent)
            return float(b)**2*(t-s)/(2*float(tau)**2)
        ref_weights=[flow_grpo_weight(t,s) if args.reference_time_weight=='flow_grpo' else
                     reference_time_weight(t,args.reference_time_weight,args.reference_time_cut,args.reference_time_exponent)
                     for t,s in zip(rt.times,rt.times[1:])]
        if rank==0:write(out/'loss_geometry.json',dict(actor_loss=args.actor_loss,
            sde_schedule=args.sde_schedule,sde_noise_exponent=args.sde_noise_exponent,
            sde_noise_mode=args.sde_noise_mode,noise_radius=args.noise_radius,
            sde_tau_per_unit_eta=unit_tau,
            sde_tau=[u*args.noise_radius for u in unit_tau] if args.sde_noise_mode=='constant' else None,
            final_direction_normalization=args.direction_normalization,latent_coordinates=int(np.prod(rt.shape)),
            reward_normalization=args.reward_normalization,time_weight_mode=args.time_weight_mode,
            times=rt.times[:-1],time_weights=base_weights.tolist(),coordinate_reduction='sum' if args.actor_loss=='fdfo_first_order' else 'mean',
            velocity_gradient_coefficient=2. if args.actor_loss=='fdfo_first_order' else 1./np.prod(rt.shape),
            sample_reduction='global mean over main states only',
            trainable_steps=trainable_steps,
            deterministic_steps=[q for q in range(len(rt.times)-1) if q not in trainable_set],
            paper_timestep_fraction=args.paper_timestep_fraction if args.method=='paper' else None,
            paper_max_btau=args.paper_max_btau if args.method=='paper' else None,
            paper_kl_convention=args.paper_kl_convention if args.method=='paper' else None,
            paper_kl_scale=paper_kl_scale if args.method=='paper' else None))
        opt=torch.optim.AdamW(rt.params,lr=args.lr,betas=(.9,.999),eps=args.adam_eps,weight_decay=1e-4)
        # Residual-error curriculum: draw prompts toward the ones still failing. The table lives in the run
        # dir so a --resume keeps it; with --resume from another run its table is copied if present.
        residual=None
        if args.prompt_weighting in ('residual','residual_mixed'):
            residual=PromptResidualTable(prompts,args.residual_prior,args.residual_decay,args.residual_floor,
                                         tags=([meta[p_]['tag'] for p_ in prompts] if args.reward=='geneval' else None),
                                         mode=('mixed' if args.prompt_weighting=='residual_mixed' else 'success'),tag_decay=args.residual_tag_decay)
            for cand in ([out/'prompt_residual.json']+([args.resume.parent.parent.parent/'prompt_residual.json'] if args.resume else [])):
                if cand.exists():residual.load(cand);break
        tag_share_weights=None
        if args.tag_shares:
            # Stratified tag mix: --tag-shares color_attr=0.35,colors=0.2,... sets the expected share of
            # each update's prompt draws per GenEval tag; within a tag the rows keep their relative weights (duplicated
            # rows = the dataset's within-tag mixture), so w_row = share[tag] / (#rows of that tag). Multiplies the residual
            # weights when both are on. Tags absent from the list get share 0.
            if args.reward not in('geneval','multi'):raise ValueError('--tag-shares needs GenEval metadata')
            shares={k:float(v) for k,v in (kv.split('=') for kv in args.tag_shares.split(','))}
            # multi: only the GenEval rows are stratified by tag; the other sources keep weight 1 (their draw is
            # normalised within the source by select_prompt_ids_by_source).
            tags_=[meta[p_]['tag'] if (train_sources is None or train_sources[i_]=='geneval') else None for i_,p_ in enumerate(prompts)]
            counts_=collections.Counter(t_ for t_ in tags_ if t_ is not None)
            tag_share_weights=np.array([1. if t_ is None else shares.get(t_,0.)/counts_[t_] for t_ in tags_],dtype=np.float64)
            if tag_share_weights.sum()<=0:raise ValueError('--tag-shares gives every row weight 0')
        def lr_at(step):
            # Constant is the historical behaviour and stays the default. Annealing
            # exists because the plateau is diffusion dominated: the stationary reward
            # deficit of a noisy update scales with the step size, so shrinking the
            # step is what converts a random walk back into progress.
            if args.lr_schedule=='constant':return args.lr
            span=max(1,args.updates-args.lr_decay_start)
            frac=min(1.,max(0.,(step-args.lr_decay_start)/span))
            if args.lr_schedule=='linear':return args.lr+(args.lr_final-args.lr)*frac
            return args.lr_final+(args.lr-args.lr_final)*.5*(1+float(np.cos(np.pi*frac)))
        initial=get_peft_model_state_dict(rt.model)
        initial_hash=digest(initial);ema={k:v.detach().cpu().clone() for k,v in initial.items()}
        if distributed():
            hashes=[None]*world;dist.all_gather_object(hashes,initial_hash)
            assert len(set(hashes))==1,'rank initial LoRA mismatch'
        if rank==0:write(out/'initial_state.json',dict(lora_sha256=initial_hash,times=rt.times,reference_weights=ref_weights,
            trainable_params=sum(p.numel() for p in rt.params),trainable_dtype=str(rt.trainable_dtype)))
        start_step=0
        previous_totals=dict(main_trajectories=0,main_training_rows=0)
        if args.resume:
            checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
            set_peft_model_state_dict(rt.model,checkpoint['lora'])
            # --resume-weights-only keeps the LoRA (and the EMA) but starts AdamW fresh. Needed when the
            # resumed run changes the gradient scale - a different estimator, reward normalization or reward scale -
            # because the second moment carries the OLD scale and beta2=0.999 needs ~1000 updates to forget it.
            if not args.resume_weights_only:opt.load_state_dict(checkpoint['optimizer'])
            ema=checkpoint['ema'];start_step=checkpoint['step']
            previous_totals=checkpoint_training_totals(checkpoint)
            checkpoint_lr=[float(group['lr']) for group in (opt.param_groups if not args.resume_weights_only else [])] or [lr_at(start_step)]
            lr_changed=any(abs(v-lr_at(start_step))>1e-12 for v in checkpoint_lr)
            if lr_changed and not args.allow_resume_lr_change:
                raise ValueError('resumed optimizer learning rate differs from the schedule value at the '
                                 'resumed step; pass --allow-resume-lr-change to do this deliberately')
            if args.updates<=start_step:
                # --updates is the absolute final step, not a count past the checkpoint; a resume at or
                # beyond it would only re-run the initial evaluation and mark the run complete.
                raise ValueError(f'--updates {args.updates} <= resumed step {start_step}: nothing to train')
            initial_hash=digest(get_peft_model_state_dict(rt.model))
            if rank==0:write(out/'resume_state.json',dict(checkpoint=str(args.resume),step=start_step,
                lora_sha256=initial_hash,training_totals=previous_totals,optimizer_lr=args.lr,
                checkpoint_optimizer_lr=checkpoint_lr,learning_rate_changed_on_resume=lr_changed))
        def training_totals(step):
            updates=step-start_step
            return dict(main_trajectories=previous_totals['main_trajectories']+updates*update_batch.trajectories_per_update,
                        main_training_rows=previous_totals['main_training_rows']+updates*update_batch.main_rows_per_update)
        eval_policies=['current','ema'] if args.eval_policy=='both' else [args.eval_policy]
        eval_guidances=args.eval_cfg or [1.]
        elapsed_train=0.;elapsed_eval=0.;elapsed_checkpoint=0.;all_eval=[];last_metrics=None
        def evaluate(step):
            # One record per requested policy, all on the same ruler: identical prompt
            # sharding, eval_steps, eval_chunk and seeds, so the only thing that differs
            # between the rows is which weights produced them.
            nonlocal elapsed_eval
            produced=[]
            for policy in eval_policies:
                if policy=='ema':
                    live={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(rt.model).items()}
                    live_hash=digest(live);set_peft_model_state_dict(rt.model,ema)
                for guidance in eval_guidances:
                    sync();tic=time.monotonic();torch.cuda.reset_peak_memory_stats()
                    mine=list(range(rank,len(evalprompts),world))
                    # Encode this rank's whole slice once. Preparing per prompt would move the
                    # text encoders, T5-XXL included, across PCIe for every evaluation prompt.
                    rt.prepare([evalprompts[i] for i in mine])
                    pairs=[(i,31_000_000_000+args.eval_seed*100000+i*100+j)
                           for i in mine for j in range(args.eval_seeds)]
                    scores=rt.evaluate_rows([(evalprompts[i],s) for i,s in pairs],args.eval_steps,
                                            args.eval_chunk,guidance=guidance,grid_mix=args.eval_grid_mix)
                    rows=[dict(prompt_id=i,seed=s,reward=r) for (i,s),r in zip(pairs,scores)]
                    if rt.last_eval_details is not None:
                        for row,detail in zip(rows,rt.last_eval_details):row.update(detail)
                    if distributed():
                        parts=[None]*world;dist.all_gather_object(parts,rows);rows=[r for p in parts for r in p]
                    seconds=allmax(time.monotonic()-tic,rt.device);elapsed_eval+=seconds
                    record=dict(step=step,mean=float(np.mean([r['reward'] for r in rows])),
                        display_mean=float(np.mean([r['reward'] for r in rows]))/(26 if args.reward=='pickscore' else 1),
                        reward=args.reward,**(multi_summary(rows) if args.reward=='multi' else reward_summary(rows)),rows=rows,
                        train_seconds=elapsed_train,eval_seconds=seconds,cumulative_eval_seconds=elapsed_eval,
                        wall_seconds=time.monotonic()-started,
                        evaluation_policy='current LoRA' if policy=='current' else f'EMA decay {args.ema_decay}',
                        eval_policy=policy,eval_cfg=guidance,eval_steps=args.eval_steps,eval_grid_mix=args.eval_grid_mix,
                        peak_eval_gb=torch.cuda.max_memory_allocated()/1e9,
                        **training_totals(step))
                    all_eval.append(record);produced.append(record)
                    if rank==0:
                        append(out/'eval_metrics.jsonl',record)
                        print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
                    status('evaluation_complete',optimizer_steps=step,eval_mean=record['mean'],
                           eval_policy=policy,eval_cfg=guidance)
                if policy=='ema':
                    set_peft_model_state_dict(rt.model,live)
                    if digest(get_peft_model_state_dict(rt.model))!=live_hash:
                        raise RuntimeError('live LoRA not restored exactly after the EMA evaluation')
            return produced[0]
        def checkpoint(step,light=False):
            """Full state, or LoRA only.

The optimizer is 143 MB against the LoRA's 72 MB - AdamW keeps two moments - so a
light checkpoint is a quarter of a full one and can be written every update. What
that buys is SWA windows at update granularity instead of multiples of save_every.
A light checkpoint cannot be resumed from: --resume loads the optimizer
unconditionally, which is why the full ones keep their own cadence.
"""
            nonlocal elapsed_checkpoint
            tic=time.monotonic()
            if rank==0:
                root=out/'checkpoints'/f'checkpoint-{step:06d}'
                root.mkdir(parents=True,exist_ok=True)
                lora={k:v.detach().cpu() for k,v in get_peft_model_state_dict(rt.model).items()}
                if not light:
                    torch.save(dict(lora=lora,ema=ema,optimizer=opt.state_dict(),step=step,config=cfg,
                                    training_totals=training_totals(step)),root/'state.pt')
                    torch.save(ema,root/'lora_ema_state.pt')
                    rt.model.peft_config['default'].save_pretrained(root)
                torch.save(lora,root/'lora_state.pt')
                write(root/'progress.json',dict(step=step,lora_sha256=digest(lora),train_seconds=elapsed_train,
                                               light=light,**training_totals(step)))
            if distributed():dist.barrier()
            elapsed_checkpoint+=time.monotonic()-tic
        # The EMA recursion starts at the initial weights, so offline reconstruction of
        # any decay needs them on disk too. Light, so it costs 72 MB.
        if args.save_lora_every:checkpoint(start_step,light=True)
        baseline=evaluate(start_step)
        if args.eval_only:
            summary=dict(status='complete',mode='eval_only',start_step=start_step,world_size=world,
                eval_steps=args.eval_steps,eval_policies=eval_policies,
                checkpoint=str(args.resume) if args.resume else None,
                wall_seconds=time.monotonic()-started,eval_seconds=elapsed_eval,
                evaluations=[{k:v for k,v in r.items() if k!='rows'} for r in all_eval])
            if rank==0:write(out/'summary.json',summary)
            status('complete',optimizer_steps=start_step,stop_reason='eval_only',
                   final_reward=all_eval[-1]['mean'])
            return
        stop_reason='maximum_updates';last_actor_norm=0.;fm_scale=0.
        for step in range(start_step+1,args.updates+1):
            # Decided on rank 0 and shared: per-rank checks can land on different boundaries,
            # and the ranks then disagree about which collectives come next.
            halt=allmax((1. if (out/'STOP_REQUEST.json').exists() else
                         2. if time.monotonic()-started>args.max_hours*3600 else 0.) if rank==0 else 0.,rt.device)
            if halt==1.:stop_reason='stop_request_at_update_boundary';break
            if halt==2.:stop_reason='maximum_hours';break
            sync();tic=time.monotonic();opt.zero_grad(set_to_none=True)
            rollout_seconds=estimate_seconds=backward_seconds=score_seconds=0.
            policy_loss=kl_loss=0.;reward_values=[];sde_values=[];replay=0.;ratio_error=0.;rows_used=0
            gen_samples=score_count=train_samples=0;directions_rms=[];state_kept=[];rms_by_time=np.zeros(len(rt.times)-1);group_class_fractions=None;nonzero_rows_local=None;padded_rows_local=0;gsize=args.group_size;extra_local=mixed_local=0;trained_fail_local=trained_succ_local=0
            gain_scales=[];step_weights=[];informative_groups=[];ode_reward_values=[];adjacent_zero=[0,0];gain_mean_by_q=None;peak_phase={}
            kappas=[];scale_used=None;fm_stash=None;fm_norm=0.;fm_count=0;fm_loss_total=0.;actor_norm=None
            # --train-state-stride K (40-step rollouts need not train every state): this update trains only
            # the states q = phase (mod K); the phase is a function of (seed, step) so all ranks agree without a collective and
            # every state is trained once per K updates on average. The loss normaliser divides by the TRAINED step count so
            # the gradient stays the mean over trained (row, q) pairs - same magnitude as training every state.
            state_phase=(args.seed*1000003+step*7919)%args.train_state_stride if args.train_state_stride>1 else 0
            step_set=({q for q in trainable_set if q%args.train_state_stride==state_phase}
                      if args.train_state_stride>1 and args.rollout_mode=='ode_group' else trainable_set)
            if args.reward=='multi':
                # Source-stratified draw (fixed group count per source, shuffled so every rank gets a mix of sources); the
                # residual curriculum and the GenEval tag shares enter as within-source row weights.
                rw_=None
                if residual is not None or tag_share_weights is not None:
                    rw_=(residual.weights() if residual is not None else np.ones(len(prompts)))*(tag_share_weights if tag_share_weights is not None else 1.)
                pids=select_prompt_ids_by_source(args.seed,step,train_sources,source_shares,args.groups_per_update,rw_)
            elif residual is not None or tag_share_weights is not None:
                w_=(residual.weights() if residual is not None else np.ones(len(prompts)))*(tag_share_weights if tag_share_weights is not None else 1.)
                pids=select_prompt_ids_weighted(args.seed,step,w_,args.groups_per_update)
            else:pids=select_prompt_ids(args.seed,step,len(prompts),args.groups_per_update,args.unique_prompts_per_update)
            if not (args.prepare_local_prompts and args.batching=='rows'):rt.prepare([prompts[i] for i in pids])
            groups=[] if args.batching=='rows' else list(enumerate(pids))
            if args.batching=='rows':
                # One flat shard per rank. Batches may mix prompts, so the microbatch
                # is a free choice instead of group_size/world, and the trajectory
                # count, the prompt count and the batch width stop constraining
                # one another.
                allrows=build_rows(pids,args.group_size);rows=shard_rows(allrows,world,rank,args.shard_mode)
                texts=[prompts[pid] for _,pid,_ in rows];total=len(rows)
                # 288 prompts x 1 trajectory: every rank would otherwise run T5-XXL over all 288 prompts although it only
                # conditions on its own 36 rows.
                if args.prepare_local_prompts:rt.prepare(list(dict.fromkeys(texts)))
                rt.model.eval();sync();t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
                sampler=(sample_rows_paper if args.method=='paper' else
                         sample_rows_ode if args.rollout_mode=='ode_group' else sample_rows)
                bank,cost=sampler(rt,rows,prompts,args,step,args.train_microbatch)
                peak_phase['rollout']=torch.cuda.max_memory_allocated()/1e9;torch.cuda.reset_peak_memory_stats()
                if args.reward=='multi' and (args.quality_mode in ('group_floor','task_floor') or args.task_exact_bonus or quality_margin):
                    # The quality floor of a rule row is its own group's mean q (stepwise_quality.compose_multi_rewards).
                    # One composed reward per SCORED STATE: the ODE sampler gives one column, the SDE sampler eleven
                    # (stepwise_rows.sample_rows), and the estimator reads adjacent differences of bank['r'].
                    bank['r']=compose_multi_states(bank['src'],bank['r_task'],bank['q_all'],args.pickapic_scale,args.quality_lambda,
                                                   args.quality_mode,rt.quality_ref,groups=group_slices(rows,args.group_size),
                                                   exact_bonus=args.task_exact_bonus,margin=quality_margin,task_lambda=args.task_lambda)
                sync();rollout_seconds+=time.monotonic()-t0;score_seconds+=cost['score_seconds']
                gen_samples+=cost['forward_samples'];score_count+=cost['score_samples']
                reward_values.extend(bank['r'][:,-1].tolist());sde_values.append(cost['innovation_rms_mean'])
                # r[:,0] is the ODE completion of the initial noise, the on-policy image the
                # 10-step deterministic sampler would give; r[:,-1] is the realized SDE
                # endpoint. Their gap is what the injected noise costs in reward, and the
                # exact-zero share of adjacent gains is how much of the estimator's input a
                # discrete reward leaves empty.
                ode_reward_values.extend(bank['r'][:,0].tolist())
                gains=bank['r'][:,1:]-bank['r'][:,:-1]
                adjacent_zero[0]+=int((gains==0).sum());adjacent_zero[1]+=gains.numel()
                t0=time.monotonic();ahead=args.direction_index=='next';rewards=bank['r'];advantages=None
                if args.method=='paper':
                    # Eq. (5) normalises one prompt group jointly over N x T. A group is
                    # spread across every rank here, so the mean and std come from the
                    # gathered rewards and are then applied to this rank's rows - which
                    # is joint_gain_advantages restricted to the local shard.
                    order=gathered_order(allrows,world,args.shard_mode)
                    allr=gather(bank['r'].to(rt.device)).cpu()
                    advantages=torch.zeros(len(rows),len(rt.times)-1)
                    for g in range(len(pids)):
                        gmean,gstd=group_gain_statistics(allr[group_positions(order,g)])
                        mine=group_positions(rows,g)
                        if not mine:continue
                        r=bank['r'][mine].double()
                        advantages[mine]=((r[:,1:]-r[:,:-1]-gmean)/gstd.clamp_min(1e-8)).float()
                elif reward_scale is not None:
                    # No baseline and no group statistic: the regression consumes reward differences, so only
                    # a unit-setting scalar remains - a constant, or the std of this update's gathered final rewards.
                    scale_used=(batch_reward_scale(gather(bank['r'][:,-1].contiguous().to(rt.device)).cpu())
                                if reward_scale=='batch_std' else reward_scale)
                    rewards=bank['r']/scale_used
                elif args.reward_normalization!='none':
                    # A prompt group is spread over every rank here, so its gain
                    # statistic has to come from the gathered update, not the shard.
                    order=gathered_order(allrows,world,args.shard_mode)
                    allr=gather(bank['r'].to(rt.device)).cpu();rewards=bank['r'].clone()
                    gain_mean_by_q=step_gain_statistics(allr)[0].tolist()
                    if normalizer.needs_update_statistics:normalizer.begin_update(allr)
                    squares=allsq=None
                    if normalizer.needs_step_squares:
                        squares=endpoint_step_squares(bank['y']);allsq=gather(squares.to(rt.device)).cpu()
                    for g in range(len(pids)):
                        # Every rank walks all groups in this same order, so the running
                        # estimate stays identical across ranks; a rank holding no row of
                        # the group still advances it. Groups whose gains are all equal
                        # (discrete rewards) are skipped by the running update.
                        mine=group_positions(rows,g);whole=group_positions(order,g)
                        out_rewards,scale,informative=normalizer(bank['r'][mine],allr[whole],step,
                            None if squares is None else squares[mine],None if allsq is None else allsq[whole])
                        if mine:rewards[mine]=out_rewards
                        informative_groups.append(informative)
                        gain_scales.append(scale)
                        if normalizer.last_kappa is not None:kappas.append(normalizer.last_kappa)
                train_idx=list(range(len(rows)))
                if args.rollout_mode=='ode_group' and not args.cloud_size:
                    # Group classes for the plateau diagnosis: solved = every endpoint at the cap, stuck = all
                    # equal but below it (no within-group contrast -> never pushed), mixed = the groups that carry signal.
                    gs=group_slices(rows,args.group_size);gr=(bank['r_task'] if 'r_task' in bank else bank['r_ocr'] if 'r_ocr' in bank else bank['r'])[:,0]
                    # multi: a Pick-a-Pic group (continuous reward, no cap) is its own class [.., .., .., continuous].
                    cls=torch.zeros(len(gs),4,dtype=torch.float64);gpid=torch.zeros(len(gs),dtype=torch.long);gsucc=torch.zeros(len(gs),dtype=torch.float64)
                    for k,(g,sl) in enumerate(gs):
                        if 'src' in bank and int(bank['src'][sl.start])==0:
                            cls[k,3]=1.;gpid[k]=pids[g];gsucc[k]=0.;continue
                        r_=gr[sl].double();same=bool((r_.max()-r_.min())<1e-6)
                        cls[k,0 if (same and float(r_.min())>=.99) else 1 if same else 2]=1.
                        gpid[k]=pids[g];gsucc[k]=float((r_>=.99).double().mean())
                    allcls=gather(cls.to(rt.device)).cpu();allpid=gather(gpid.to(rt.device)).cpu();allsucc=gather(gsucc.to(rt.device)).cpu()
                    group_class_fractions=(allcls.mean(0)).tolist();mixed_local=int(cls[:,2].sum())   # [solved, stuck, mixed, continuous]
                    if residual is not None:
                        # continuous (Pick-a-Pic) groups have no success rate: leave their prompts at the prior
                        keep_=(allcls[:,3]==0)
                        residual.update(allpid[keep_].tolist(),allsucc[keep_].tolist())
                        if rank==0:residual.save(out/'prompt_residual.json')
                    if args.mixed_extra_k:
                        # Adaptive K: a second ODE pass of --mixed-extra-k trajectories for
                        # every group of THIS rank that showed reward contrast in the first pass (the block shard keeps a
                        # group on one rank). Solved / stuck groups get nothing. The extra rows join their group so the
                        # sibling fit sees K + extra endpoints and the low-row selection has more failures to choose
                        # from; per-rank rollout cost grows with the rank's mixed-group count. Train-reward statistics
                        # above keep the first pass only; the extras are logged separately.
                        extra_rows_=[(g,rows[sl.start][1],j) for k,(g,sl) in enumerate(gs) if cls[k,2]>0
                                     for j in range(args.group_size,args.group_size+args.mixed_extra_k)]
                        extra_local=len(extra_rows_)
                        if extra_rows_:
                            sync();t1=time.monotonic()
                            bank2,cost2=sample_rows_ode(rt,extra_rows_,prompts,args,step,args.train_microbatch)
                            sync();rollout_seconds+=time.monotonic()-t1;score_seconds+=cost2['score_seconds']
                            gen_samples+=cost2['forward_samples'];score_count+=cost2['score_samples']
                            merged={k_:torch.cat([bank[k_],bank2[k_]]) for k_ in bank}
                            order_=[];new_rows=[];pos=len(rows)
                            for k,(g,sl) in enumerate(gs):
                                order_.extend(range(sl.start,sl.stop));new_rows.extend(rows[sl])
                                if cls[k,2]>0:
                                    order_.extend(range(pos,pos+args.mixed_extra_k));new_rows.extend(extra_rows_[pos-len(rows):pos-len(rows)+args.mixed_extra_k]);pos+=args.mixed_extra_k
                            assert pos==len(rows)+len(extra_rows_)==len(order_),'extra-row merge mismatch'
                            bank={k_:v[order_] for k_,v in merged.items()};rows=new_rows;texts=[prompts[pid] for _,pid,_ in rows];total=len(rows)
                            if args.reward=='multi' and (args.quality_mode in ('group_floor','task_floor') or args.task_exact_bonus or quality_margin):
                                bank['r']=compose_multi_states(bank['src'],bank['r_task'],bank['q_all'],args.pickapic_scale,args.quality_lambda,
                                                               args.quality_mode,rt.quality_ref,groups=group_slices(rows,None),
                                                               exact_bonus=args.task_exact_bonus,margin=quality_margin,task_lambda=args.task_lambda)
                            rewards=bank['r'] if scale_used is None else bank['r']/scale_used
                            del merged,bank2
                        gsize=None
                if args.fit_reward_floor is not None and args.method!='paper':
                    # Regression + positive-target rule: the rewards the direction fits see are
                    # clipped from below, so GenEval clutter (negative count credit) and an empty image (0) are level and
                    # "toward the emptier completion" is no longer a slope. State selection / logging use the raw bank['r'].
                    rewards=rewards.clamp_min(args.fit_reward_floor)
                if args.method=='paper':directions=None
                elif args.rollout_mode=='ode_group':
                    # Sibling contrast (stepwise_group_fit.sibling_directions): the direction of a trained trajectory is
                    # one ridge over the other K-1 ODE endpoints of its prompt group, repeated along T. Only the first
                    # --train-per-group trajectories of each group carry actor rows; the rest supply endpoints.
                    per=args.train_per_group or args.group_size
                    train_idx=[];parts=[]
                    # Update-wide reward std for the NFT-style advantages of --direction-mode anchor/both (NFT's global_std).
                    gstd_all=float(gather(bank['r'].reshape(-1).to(rt.device)).double().std()) if args.direction_mode!='sibling' else None
                    sel_gen=torch.Generator().manual_seed(args.seed*1000003+step*7919+rank)
                    for g,sl in group_slices(rows,gsize):
                        if args.cloud_size:
                            # ODE-cloud group: M mains, each followed by its C perturbed members; only mains carry actor rows.
                            width=1+args.cloud_size;mains=args.group_size//width
                            train_idx.extend(sl.start+m*width for m in range(mains))
                            parts.append(cloud_directions(bank['y'][sl,0].to(rt.device),rewards[sl,0].to(rt.device),mains,
                                                          args.cloud_size,args.ridge,len(rt.times)-1,args.cloud_pool_mains).cpu())
                        else:
                            yg=bank['y'][sl,0].to(rt.device);rg=rewards[sl,0].to(rt.device)
                            # Mixed reward (OCR + PickScore / multi): the task component decides who counts as a failure; the mixed
                            # value orders rows. A Pick-a-Pic group (multi, continuous reward) has no failure rule.
                            has_task='r_task' in bank or 'r_ocr' in bank
                            raw_r=(bank['r_task'] if 'r_task' in bank else bank['r_ocr'] if 'r_ocr' in bank else bank['r'])[sl,0]
                            rule_group=not ('src' in bank and int(bank['src'][sl.start])==0)
                            tidx=select_trained(rg,per,args.train_select,sel_gen,args.train_max_reward if rule_group else None,
                                                raw_r.to(rt.device) if has_task else None)
                            train_idx.extend(sl.start+i for i in tidx)
                            if rule_group:
                                trained_fail_local+=sum(float(raw_r[i])<.99 for i in tidx);trained_succ_local+=sum(float(raw_r[i])>=.99 for i in tidx)
                            fam=[]
                            if args.direction_mode in('sibling','both'):
                                fam.append(sibling_directions(yg,rg,tidx,args.ridge,len(rt.times)-1,args.sibling_centre))
                            if args.direction_mode in('anchor','both'):
                                # NFT's one-sample value gradient on the ODE's own states (stepwise_group_fit.anchor_directions).
                                da=anchor_directions(bank['x'][sl].to(rt.device),bank['old'][sl].to(rt.device),rt.times,yg,rg,tidx,
                                                     gstd_all,args.anchor_adv_clip)
                                fam.append(da*args.anchor_mix if args.direction_mode=='both' else da)
                            if args.direction_mode=='both':
                                # Each family enters at its own group rms so --anchor-mix is a ratio, not a unit mismatch.
                                fam=[f/f.double().square().mean().sqrt().clamp_min(1e-30).float() if float(f.abs().max())>0 else f for f in fam]
                            d=sum(fam)
                            # Unit-rms anchor rows are far above the sibling ridge's raw scale; AdamW (eps 1e-15) is scale
                            # invariant but --grad-clip is not, so bring the gradient back to the sibling directions' scale.
                            if args.direction_mode!='sibling':d=d*args.anchor_scale
                            parts.append(d.cpu())
                    directions=torch.cat(parts)
                elif args.fit_scope=='nearest_success_group':
                    # One sharp success target per (trajectory, state) from the whole prompt group (block shard keeps the
                    # group on this rank); own-trajectory completions are usually the nearest, siblings fill in otherwise.
                    directions=torch.empty(len(rows),len(rt.times)-1,*bank['y'].shape[2:])
                    for g,sl in group_slices(rows,args.group_size):
                        directions[sl]=group_nearest_success_directions(bank['y'][sl].to(rt.device),rewards[sl].to(rt.device),args.state_cap).cpu()
                elif args.fit_scope=='pool_head':
                    # One ridge per (trajectory, q) over every head endpoint of the prompt group; the group
                    # is contiguous on this rank (block shard), so no gather is needed. Fitted on the GPU:
                    # 6 x 8 rows of 65536 per fit is nothing there and seconds per group on the CPU.
                    directions=torch.empty(len(rows),len(rt.times)-1,*bank['y'].shape[2:])
                    for g,sl in group_slices(rows,args.group_size):
                        directions[sl]=pooled_group_directions(bank['y'][sl].to(rt.device),rewards[sl].to(rt.device),
                                                               args.fit_head_rows,args.ridge,args.pool_nearest).cpu()
                    if args.direction_normalization=='state_rms':
                        scale=directions.flatten(2).square().mean(2).sqrt()
                        directions/=scale.clamp_min(1e-30).reshape(*scale.shape,*([1]*(directions.ndim-2)))
                else:directions=torch.stack(
                    [trajectory_window_directions(y,r,args.fit_scope,args.fit_width,
                                           args.future_cap,args.ridge,
                                           normalize=args.direction_normalization=='state_rms',
                                           include_final=ahead,head_rows=args.fit_head_rows,target_min=args.target_min)
                     for y,r in zip(bank['y'],rewards)])
                if directions is not None and ahead:directions=directions[:,1:]
                if directions is not None and args.row_normalize:
                    # On a discrete reward a secant dr/|dy| diverges as the chord shrinks (a step has
                    # no slope); the smoothed value gradient is bounded by dr/l. One flip = one vote: unit rms per (row, q),
                    # NFT's adaptive per-sample weighting. Applied at the estimator level so the FDFO loss keeps its contract.
                    rr_=directions.double().flatten(2).square().mean(2).sqrt()
                    directions=(3e-5*directions.double()/rr_.clamp_min(1e-30).reshape(*rr_.shape,*([1]*(directions.ndim-2)))).float()*(rr_>0).float().reshape(*rr_.shape,*([1]*(directions.ndim-2)))
                if directions is not None and args.state_select!='all' and args.rollout_mode=='sde':
                    # Train only the states whose own ODE completion is low (below the best
                    # completion of the same trajectory, or below the cap); states already at a high completion get no
                    # push, so the "away from the mean" sharpening force of the self term never fires.
                    nq=directions.shape[1];rr=bank['r'][:,1:nq+1] if ahead else bank['r'][:,:nq]
                    keep=(rr<bank['r'].max(1,keepdim=True).values-1e-6) if args.state_select=='below_max' else (rr<args.state_cap)
                    directions=directions*keep.float().reshape(*keep.shape,*([1]*(directions.ndim-2)))
                    state_kept.append(float(keep.float().mean()))
                weights=base_weights
                if directions is not None:
                    if args.direction_normalization=='group_rms':
                        energy=gather(directions.double().square().mean().reshape(1).to(rt.device)).mean().sqrt()
                        directions/=max(float(energy),1e-30)
                    row_rms=directions.double().flatten(2).square().mean(2).sqrt()
                    directions_rms.extend(row_rms.flatten().tolist());rms_by_time+=row_rms.sum(0).numpy()
                    if args.time_weight_mode=='inverse_rms':
                        sq=gather(row_rms.square().sum(0).to(rt.device)).view(world,-1).sum(0).cpu()
                        weights=inverse_rms_time_weights(sq,torch.full_like(sq,float(update_batch.trajectories_per_update)))
                        step_weights.append(weights.tolist())
                sync();estimate_seconds+=time.monotonic()-t0
                rt.model.train();t0=time.monotonic()
                if args.fm_weight:
                    # Advantage-weighted flow matching on the scored endpoints (stepwise_advantage_fm): run first, so its
                    # gradient can be measured on its own, averaged over ranks and stashed; it is added back to the
                    # actor gradient after that one is averaged, rescaled to fm_weight times the actor gradient norm.
                    slices=group_slices(rows,gsize)
                    gstd=float(gather(bank['r'].reshape(-1).to(rt.device)).double().std())
                    fm_w=endpoint_advantages(bank['r'],slices,gstd,args.fm_adv_clip)
                    if args.fm_mode=='positive':fm_w=fm_w.clamp_min(0.)
                    gen=torch.Generator().manual_seed(args.seed*1000003+step*7919+rank)
                    picks=select_endpoints(fm_w,slices,args.fm_per_group,gen)
                    fm_count=len(picks);fm_local=0.
                    for lo in range(0,len(picks),args.train_microbatch):
                        part=picks[lo:lo+args.train_microbatch]
                        for _ in range(args.fm_times):
                            tq=int(torch.randint(len(rt.times)-1,(1,),generator=gen));t=float(rt.times[tq])
                            x0=torch.stack([bank['y'][i,k] for i,k in part]).to(rt.device)
                            eps=torch.randn(x0.shape,generator=gen).to(rt.device)
                            x_t=(1.-t)*x0+t*eps
                            pred=rt.velocity_rows(x_t,t,[texts[i] for i,_ in part])
                            fl=advantage_fm_loss(pred,x_t,t,x0,torch.tensor([float(fm_w[i,k]) for i,k in part],device=rt.device))
                            fl=fl*len(part)/max(1,len(picks)*args.fm_times)
                            if not torch.isfinite(fl):raise RuntimeError('nonfinite fm loss')
                            fl.backward();fm_local+=float(fl.detach())
                    fm_stash=[]
                    for p in rt.params:
                        g=p.grad if p.grad is not None else torch.zeros_like(p)
                        if distributed():dist.all_reduce(g);g.div_(world)
                        fm_stash.append(g.detach().clone());p.grad=None
                    fm_norm=float(torch.stack([g.double().square().sum() for g in fm_stash]).sum().sqrt())
                    fm_loss_total=fm_local
                    cnt=torch.tensor([float(fm_count)],device=rt.device)
                    if distributed():dist.all_reduce(cnt)
                    fm_count=int(cnt.item())
                loss_fn=fdfo_first_order_velocity_loss if args.actor_loss=='fdfo_first_order' else velocity_ascent_loss
                act_x,act_old,act_texts,act_total=bank['x'],bank['old'],texts,total;act_norm_total=total
                if args.rollout_mode=='ode_group':
                    if args.skip_zero_rows:
                        # A row whose direction is all-zero contributes a
                        # loss of exactly 0 and a zero gradient; forwarding it only burns compute.
                        # Drop it from the actor pass; the loss keeps the ORIGINAL row count in its
                        # normaliser so the gradient magnitude is unchanged. If every row is zero the update is skipped.
                        nz=(directions.flatten(2).abs().sum(2).sum(1)>0)
                        act_norm_total=len(train_idx)
                        train_idx=[i for i,k in zip(train_idx,nz.tolist()) if k];directions=directions[nz]
                        nonzero_rows_local=len(train_idx)
                        # Crash guard: a ragged last microbatch does not replay the rollout's bf16 forward
                        # bitwise, only the full microbatch shape does.
                        # Pad the kept rows up to a multiple of the microbatch
                        # with repeats of kept rows carrying an all-zero direction: same shapes as the rollout, zero loss, zero
                        # gradient, and the normaliser still divides by the ORIGINAL row count.
                        pad=(-len(train_idx))%args.train_microbatch if train_idx else 0
                        if pad:
                            train_idx=train_idx+[train_idx[k%len(train_idx)] for k in range(pad)]
                            directions=torch.cat([directions,torch.zeros((pad,)+tuple(directions.shape[1:]),dtype=directions.dtype,device=directions.device)])
                        padded_rows_local=pad
                    else:act_norm_total=len(train_idx);nonzero_rows_local=len(train_idx);padded_rows_local=0
                    act_x=bank['x'][train_idx];act_old=bank['old'][train_idx];act_texts=[texts[i] for i in train_idx];act_total=len(train_idx)
                row_weight=((lambda n:n/(max(act_norm_total,1)*len(step_set))) if args.rollout_mode=='ode_group'
                            else update_batch.loss_weight)
                for q,t in enumerate(rt.times[:-1]):
                    if q not in step_set:continue
                    for b0 in range(0,act_total,args.train_microbatch):
                        sl=slice(b0,min(b0+args.train_microbatch,act_total));n=sl.stop-sl.start
                        x=act_x[sl,q].to(rt.device);old=act_old[sl,q].to(rt.device)
                        v=rt.velocity_rows(x,t,act_texts[sl])
                        replay=max(replay,float((v.detach()-old).double().square().mean().sqrt()))
                        kl=torch.zeros((),device=rt.device)
                        if args.method=='paper':
                            # Identical to the prompt-major branch; only the conditioning
                            # is gathered per row instead of shared by the whole batch.
                            tau=bank['tau'][sl,q].to(rt.device);bq=bank['b'][sl,q].to(rt.device)
                            old_mu=bank['mean'][sl,q].to(rt.device);mu=old_mu+bq*(v-old)
                            child=bank['child'][sl,q].to(rt.device)
                            if not (tau>0).all():
                                raise ValueError('DDIM deterministic edge requires an explicitly separate boundary treatment')
                            logp=gaussian_logprob(child,mu,tau);oldp=gaussian_logprob(child,old_mu,tau)
                            pl,ratio=grpo_loss(logp,oldp,advantages[sl,q].to(rt.device),args.clip_range)
                            ratio_error=max(ratio_error,float((ratio.detach()-1).abs().max()))
                            if args.reference_beta:
                                with torch.no_grad():
                                    rt.model.disable_adapters();ref=rt.velocity_rows(x,t,act_texts[sl]);rt.model.enable_adapters()
                                refmu=old_mu+bq*(ref-old)
                                kl=gaussian_kl(mu,refmu,tau)*args.reference_beta*paper_kl_scale[q]
                        else:
                            pl=loss_fn(v,old,directions[sl,q].to(rt.device),float(weights[q])) if policy_on(t) else v.sum()*0.
                            # Zero-weight steps (early/late5 shapes) skip the base forward: it is pure extra cost.
                            if args.reference_beta and ref_weights[q]:
                                with torch.no_grad():
                                    rt.model.disable_adapters();ref=rt.velocity_rows(x,t,act_texts[sl]);rt.model.enable_adapters()
                                kl=args.reference_beta*ref_weights[q]*(v-ref).square().mean()
                        weight=row_weight(n)
                        loss=(pl+kl)*weight
                        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                        loss.backward();policy_loss+=float(pl.detach())*weight;kl_loss+=float(kl.detach())*weight
                        rows_used+=n;train_samples+=n
                    status('training',optimizer_steps=step-1,next_step=step,completed_main_times=q+1,
                           main_times=len(rt.times)-1,local_main_rows=rows_used)
                sync();backward_seconds+=time.monotonic()-t0;peak_phase['backward']=torch.cuda.max_memory_allocated()/1e9
            for group,pid in groups:
                prompt=prompts[pid];rt.model.eval();sync();t0=time.monotonic()
                bank,cost=sample_group(rt,prompt,args,step,group,rank,world)
                sync();rollout_seconds+=time.monotonic()-t0;score_seconds+=cost['score_seconds']
                gen_samples+=cost['forward_samples'];score_count+=cost['score_samples']
                reward_values.extend(bank['r'][:,-1].tolist());sde_values.append(cost['innovation_rms_mean'])
                t0=time.monotonic();batch=len(bank['x'])
                if args.method=='paper':
                    all_rewards=gather(bank['r'].to(rt.device))
                    advantages,_,_,_=joint_gain_advantages(all_rewards)
                    advantages=advantages[rank*batch:(rank+1)*batch].cpu()
                else:
                    ahead=args.direction_index=='next'
                    rewards=bank['r']
                    if args.fit_reward_floor is not None:
                        # Regression + positive-target rule: clip the reward used by the direction
                        # fit from below, so clutter (negative GenEval count credit) and an empty image (0) are level and
                        # "toward the emptier completion" is no longer a slope; state selection still sees raw rewards.
                        rewards=rewards.clamp_min(args.fit_reward_floor)
                    if reward_scale is not None:
                        scale_used=reward_scale;rewards=bank['r']/scale_used
                    if args.reward_normalization!='none':
                        # Reward-level centering cancels in a difference estimator; the
                        # baseline has to be removed from the per-step gains instead.
                        squares=endpoint_step_squares(bank['y']) if normalizer.needs_step_squares else None
                        rewards,scale,informative=normalizer(bank['r'],gather(bank['r'].to(rt.device)).cpu(),step,
                            squares,None if squares is None else gather(squares.to(rt.device)).cpu())
                        informative_groups.append(informative)
                        gain_scales.append(scale)
                        if normalizer.last_kappa is not None:kappas.append(normalizer.last_kappa)
                    directions=torch.stack([trajectory_window_directions(y,r,args.fit_scope,args.fit_width,
                                               args.future_cap,args.ridge,
                                               normalize=args.direction_normalization=='state_rms',
                                               include_final=ahead,head_rows=args.fit_head_rows)
                                            for y,r in zip(bank['y'],rewards)])
                    # Row q trains v(X_q), whose change moves X_{q+1}; 'next' pairs it
                    # with the value gradient estimated at that produced state.
                    if ahead:directions=directions[:,1:]
                    if args.state_select!='all':
                        # Train only the states whose own ODE completion is low (below the best
                        # completion of the same trajectory, or below the cap); the states already at a high completion
                        # get no push, so the "away from the mean" sharpening force of the self term never fires.
                        nq=directions.shape[1];rr=bank['r'][:,1:nq+1] if ahead else bank['r'][:,:nq]
                        keep=(rr<bank['r'].max(1,keepdim=True).values-1e-6) if args.state_select=='below_max' else (rr<args.state_cap)
                        directions=directions*keep.float().reshape(*keep.shape,*([1]*(directions.ndim-2)))
                        state_kept.append(float(keep.float().mean()))
                    if args.direction_normalization=='group_rms':
                        energy=gather(directions.double().square().mean().reshape(1).to(rt.device)).mean().sqrt()
                        directions/=max(float(energy),1e-30)
                    row_rms=directions.double().flatten(2).square().mean(2).sqrt()
                    directions_rms.extend(row_rms.flatten().tolist());rms_by_time+=row_rms.sum(0).numpy()
                    if args.time_weight_mode=='inverse_rms':
                        # Equalize the gradient energy each main time contributes, using
                        # the whole prompt group rather than this rank's shard.
                        sq=gather(row_rms.square().sum(0).to(rt.device)).view(world,-1).sum(0).cpu()
                        weights=inverse_rms_time_weights(sq,torch.full_like(sq,float(update_batch.group_size)))
                        step_weights.append(weights.tolist())
                    else:weights=base_weights
                sync();estimate_seconds+=time.monotonic()-t0
                rt.model.train();t0=time.monotonic()
                for q,t in enumerate(rt.times[:-1]):
                    if q not in trainable_set:continue
                    for b0 in range(0,batch,args.train_microbatch):
                        sl=slice(b0,min(b0+args.train_microbatch,batch));n=len(bank['x'][sl])
                        x=bank['x'][sl,q].to(rt.device);old=bank['old'][sl,q].to(rt.device)
                        v=rt.velocity(x,t,prompt)
                        replay=max(replay,float((v.detach()-old).double().square().mean().sqrt()))
                        kl=torch.zeros((),device=rt.device)
                        if args.method=='paper':
                            tau=bank['tau'][sl,q].to(rt.device);b=bank['b'][sl,q].to(rt.device)
                            # The saved old mean plus exact velocity coefficient avoids state cancellation.
                            old_mu=bank['mean'][sl,q].to(rt.device);mu=old_mu+b*(v-old)
                            child=bank['child'][sl,q].to(rt.device)
                            if not (tau>0).all():
                                raise ValueError('DDIM deterministic edge requires an explicitly separate boundary treatment')
                            logp=gaussian_logprob(child,mu,tau);oldp=gaussian_logprob(child,old_mu,tau)
                            pl,ratio=grpo_loss(logp,oldp,advantages[sl,q].to(rt.device),args.clip_range)
                            ratio_error=max(ratio_error,float((ratio.detach()-1).abs().max()))
                            if args.reference_beta:
                                with torch.no_grad():
                                    rt.model.disable_adapters();ref=rt.velocity(x,t,prompt);rt.model.enable_adapters()
                                refmu=old_mu+b*(ref-old)
                                kl=gaussian_kl(mu,refmu,tau)*args.reference_beta*paper_kl_scale[q]
                        else:
                            loss_fn=fdfo_first_order_velocity_loss if args.actor_loss=='fdfo_first_order' else velocity_ascent_loss
                            pl=loss_fn(v,old,directions[sl,q].to(rt.device),float(weights[q])) if policy_on(t) else v.sum()*0.
                            if args.reference_beta:
                                with torch.no_grad():
                                    rt.model.disable_adapters();ref=rt.velocity(x,t,prompt);rt.model.enable_adapters()
                                kl=args.reference_beta*ref_weights[q]*(v-ref).square().mean()
                        weight=update_batch.loss_weight(n)
                        loss=(pl+kl)*weight
                        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                        loss.backward();policy_loss+=float(pl.detach())*weight;kl_loss+=float(kl.detach())*weight
                        rows_used+=n;train_samples+=n
                sync();backward_seconds+=time.monotonic()-t0
                status('training',optimizer_steps=step-1,next_step=step,completed_groups=group+1,
                       groups_per_update=args.groups_per_update,local_main_rows=rows_used)
            if replay>1e-3:raise RuntimeError(f'old-policy replay mismatch {replay}')
            expected=(len(train_idx) if args.batching=='rows' and args.rollout_mode=='ode_group'
                      else update_batch.local_trajectories_per_update)*len(step_set)
            assert rows_used==expected,'auxiliary ODE rows entered optimizer or main rows missing'
            assert len(reward_values)*world==update_batch.trajectories_per_update,'main trajectory count mismatch'
            for p in rt.params:
                if p.grad is None:p.grad=torch.zeros_like(p)
                if distributed():dist.all_reduce(p.grad);p.grad.div_(world)
            if fm_stash is not None:
                actor_norm=float(torch.stack([p.grad.double().square().sum() for p in rt.params]).sum().sqrt())
                # Norm-matched mix: the FM gradient enters at fm_weight times the actor gradient's (rank-averaged)
                # norm, so fm_weight 1 gives the two terms an equal say under Adam. A zero actor gradient (no
                # direction anywhere) falls back to the last nonzero actor norm seen.
                if actor_norm>0:last_actor_norm=actor_norm
                # No actor gradient at all (e.g. --policy-time-cut 0 for an FM-only arm): the FM gradient enters raw.
                fm_scale=(args.fm_weight*(last_actor_norm/fm_norm if last_actor_norm>0 else 1.)) if fm_norm>0 else 0.
                for p,g in zip(rt.params,fm_stash):p.grad.add_(g,alpha=fm_scale)
                del fm_stash
            norm=float(torch.nn.utils.clip_grad_norm_(rt.params,args.grad_clip))
            if not np.isfinite(norm):raise RuntimeError(f'invalid gradient norm {norm}')
            empty_update=(norm==0)
            if empty_update and not args.skip_zero_rows:raise RuntimeError('invalid gradient norm 0.0')
            before_update=[p.detach().clone() for p in rt.params]
            for group in opt.param_groups:group['lr']=lr_at(step)
            if empty_update:
                # No group carried any reward contrast this update: nothing to learn, leave weights, Adam and EMA as they are.
                opt.zero_grad(set_to_none=True)
                if rank==0:print(f'step {step}: no reward contrast in any group, update skipped',flush=True)
            else:opt.step()
            sync()
            update_norm=float(torch.stack([(p.detach()-old).double().square().sum() for p,old in zip(rt.params,before_update)]).sum().sqrt())
            del before_update
            if step==start_step+1 and distributed():
                hashes=[None]*world;dist.all_gather_object(hashes,digest(get_peft_model_state_dict(rt.model)))
                assert len(set(hashes))==1,'rank parameters diverged after averaged optimizer update'
                if rank==0:write(out/'first_update_rank_sync.json',dict(step=step,world_size=world,parameter_sync_exact=True,hashes=hashes))
            # Every rank holds identical parameters after the averaged step, which the
            # first-update hash check above asserts, so every rank can carry the same
            # EMA. That is what lets --eval-policy ema run the sharded evaluation with
            # no broadcast; rank 0 is still the only writer of checkpoints.
            # Flow-GRPO's EMAModuleWrapper, when asked for it: it steps only on every
            # `update_step_interval`-th optimizer step and ramps the decay through
            # min((1+n)/(10+n), decay), which reaches the nominal value only at n=80.
            # Its `optimization_step` is 0-based and counts steps already taken, so it
            # equals this trainer's 1-based `step` minus one.
            ema_decay=min(step/(9.+step),args.ema_decay) if args.ema_warmup else args.ema_decay
            if step%args.ema_update_interval==0 and not empty_update:
                for k,v in get_peft_model_state_dict(rt.model).items():ema[k].mul_(ema_decay).add_(v.detach().cpu(),alpha=1-ema_decay)
            if args.ema_restart_every and step%args.ema_restart_every==0:
                # Centre restart: on a plateau the live weights wander far from the basin centre
                # (update noise dominates drift) while the EMA sits on it, so the live weights restart from the EMA.
                # Every rank holds the same EMA, so the reload keeps parameters bitwise identical; AdamW moments are kept (Lookahead-style).
                set_peft_model_state_dict(rt.model,ema)
            update_seconds=allmax(time.monotonic()-tic,rt.device);elapsed_train+=update_seconds
            global_rewards=gather(torch.tensor(reward_values,device=rt.device)).cpu().tolist()
            extra=dict()
            if args.fm_weight:
                extra.update(fm_grad_norm=fm_norm,actor_grad_norm=actor_norm,fm_scale=fm_scale,fm_samples=fm_count,
                             fm_loss=fm_loss_total)
            if ode_reward_values:
                global_ode=gather(torch.tensor(ode_reward_values,device=rt.device)).cpu().tolist()
                zeros=torch.tensor(adjacent_zero,device=rt.device,dtype=torch.float64)
                if distributed():dist.all_reduce(zeros)
                extra.update(train_reward_ode_mean=float(np.mean(global_ode)),
                           adjacent_gain_zero_fraction=(float(zeros[0]/zeros[1]) if float(zeros[1])>0 else None),
                           informative_group_fraction=(float(np.mean(informative_groups)) if informative_groups else None),
                           nonzero_direction_fraction=(float(np.mean([v>0 for v in directions_rms])) if directions_rms else None))
                if rt.pool is not None:
                    fails=torch.tensor([float(rt.pool.failures)],device=rt.device,dtype=torch.float64)
                    if distributed():dist.all_reduce(fails)
                    extra['reward_exceptions_cumulative']=float(fails)
            # Row counts differ per rank once zero-direction rows are skipped: sum the real (unpadded) rows over ranks.
            cc=torch.tensor([float(extra_local),float(mixed_local),float(trained_fail_local),float(trained_succ_local)],device=rt.device,dtype=torch.float64)
            if distributed():dist.all_reduce(cc)
            extra.update(extra_trajectories=int(cc[0].item()),mixed_groups=int(cc[1].item()),trained_failure_rows=int(cc[2].item()),trained_success_rows=int(cc[3].item()))
            if nonzero_rows_local is None:main_rows_global,padded_rows_global=rows_used*world,0
            else:
                rc=torch.tensor([float(nonzero_rows_local*len(step_set)),float(padded_rows_local*len(step_set))],device=rt.device,dtype=torch.float64)
                if distributed():dist.all_reduce(rc)
                main_rows_global=int(rc[0].item());padded_rows_global=int(rc[1].item())
            if rt.last_mix is not None and args.reward!='multi':
                mm=torch.tensor([rt.last_mix['ocr_mean']*rt.last_mix['n'],rt.last_mix['pickscore_mean']*rt.last_mix['n'],float(rt.last_mix['n'])],device=rt.device,dtype=torch.float64)
                if distributed():dist.all_reduce(mm)
                extra.update(train_ocr_mean=float(mm[0]/mm[2]),train_pickscore_mean=float(mm[1]/mm[2]))
            if args.reward=='multi' and 'src' in bank:
                # Per-source means over this update's trajectories (extras included): task, the three quality scores, q.
                comp=('r_task','q_pick','q_clip','q_hps','q_all');sums=torch.zeros(len(SOURCES),len(comp)+1,dtype=torch.float64)
                for si in range(len(SOURCES)):
                    m_=bank['src']==si
                    sums[si,0]=float(m_.sum())
                    for ci,c_ in enumerate(comp):sums[si,ci+1]=float(bank[c_][m_,0].double().sum())
                sums=sums.to(rt.device)
                if distributed():dist.all_reduce(sums)
                sums=sums.cpu()
                for si,name in enumerate(SOURCES):
                    n_=float(sums[si,0]);extra[f'train_{name}_rows']=int(n_)
                    if n_>0:
                        for ci,key in enumerate(('task','pickscore','clipscore','hpsv2','q')):
                            if key=='task' and name=='pickapic':continue
                            extra[f'train_{name}_{key}_mean']=float(sums[si,ci+1]/n_)
            metrics=dict(step=step,train_reward_mean=float(np.mean(global_rewards)),**extra,policy_loss=policy_loss,kl_loss=kl_loss,
                grad_norm=norm,grad_clip_scale=min(1.,args.grad_clip/(norm+1e-6)),grad_clip=args.grad_clip,parameter_update_norm=update_norm,
                learning_rate=lr_at(step),
                old_replay_rms=replay,old_ratio_max_error=ratio_error,
                update_seconds=update_seconds,train_seconds=elapsed_train,rollout_seconds=rollout_seconds,
                score_seconds=score_seconds,estimate_seconds=estimate_seconds,backward_seconds=backward_seconds,
                main_trajectories=args.groups_per_update*args.group_size,main_training_rows=main_rows_global,padded_training_rows=padded_rows_global,
                prompt_groups=args.groups_per_update,unique_prompts=len(set(pids)),prompt_ids=pids,
                auxiliary_training_rows=0,rollout_forward_samples=gen_samples*world,reward_samples=score_count*world,
                actor_training_forward_samples=train_samples*world,fresh_on_policy=True,
                reward_normalization=args.reward_normalization,time_weight_mode=args.time_weight_mode,
                gain_scale_mean=float(np.mean(gain_scales)) if gain_scales else None,
                running_gain_std=normalizer.running,
                running_gain_step_profile=(normalizer.step_profile(normalizer.running_steps).tolist()
                                           if normalizer.running_steps is not None else None),
                gain_mean_by_q=gain_mean_by_q,
                curvature_kappa_mean=float(np.mean(kappas)) if kappas else None,
                reward_scale_used=scale_used,adam_eps=args.adam_eps,
                applied_time_weights=(np.mean(step_weights,0).tolist() if step_weights
                                      else base_weights.tolist()),
                direction_rms_mean=float(np.mean(directions_rms)) if directions_rms else None,
                state_kept_fraction=(float(np.mean(state_kept)) if state_kept else None),
                group_class_fractions=group_class_fractions,residual_table=(residual.summary() if residual is not None else None),
                direction_rms_by_time=(rms_by_time/len(reward_values)).tolist() if directions_rms else None,
                train_state_stride=args.train_state_stride,trained_state_phase=state_phase,trained_states=sorted(step_set),
                direction_diagnostics_scope='rank0 trajectories' if rank==0 else f'rank{rank} trajectories',
                cumulative_main_trajectories=training_totals(step)['main_trajectories'],
                cumulative_main_rows=training_totals(step)['main_training_rows'],
                peak_cuda_gb=max(peak_phase.values()) if peak_phase else torch.cuda.max_memory_allocated()/1e9,
                peak_phase_gb=peak_phase)
            last_metrics=metrics
            if rank==0:append(out/'metrics.jsonl',metrics);print(json.dumps(metrics),flush=True)
            status('running',optimizer_steps=step,train_reward_mean=metrics['train_reward_mean'],train_seconds=elapsed_train)
            if step%args.eval_every==0 or step==args.updates or (args.eval_first and step==start_step+1):evaluate(step)
            full=step%args.save_every==0 or step==args.updates or (args.save_first and step==start_step+1)
            if full:checkpoint(step)
            elif args.save_lora_every and step%args.save_lora_every==0:checkpoint(step,light=True)
        actual_step=last_metrics['step'] if last_metrics else start_step
        if all_eval[-1]['step']!=actual_step:evaluate(actual_step)
        # rank 0 creates the directory before it finishes writing, so a per-rank existence check races it:
        # a rank that sees the half-written directory skips checkpoint() and its barrier, and rank 0 waits in it.
        # state.pt, not the directory: with --save-lora-every every step has a LoRA-only directory, and a stop request
        # between full checkpoints would leave no resumable state at the stop step.
        saved=allmax(float((out/'checkpoints'/f'checkpoint-{actual_step:06d}'/'state.pt').exists()) if rank==0 else 0.,rt.device)
        if actual_step and not saved:checkpoint(actual_step)
        change_hash=digest(get_peft_model_state_dict(rt.model))
        if actual_step>start_step and initial_hash==change_hash:raise RuntimeError('LoRA unchanged')
        summary=dict(status='complete',method=args.method,actor_loss=args.actor_loss,
            direction_normalization=args.direction_normalization,time_exponent=args.time_exponent,lr=args.lr,
            optimizer_steps=actual_step,start_step=start_step,updates_in_this_run=actual_step-start_step,stop_reason=stop_reason,
            baseline_reward=baseline['mean'],final_reward=all_eval[-1]['mean'],delta=all_eval[-1]['mean']-baseline['mean'],
            train_seconds=elapsed_train,eval_seconds=elapsed_eval,checkpoint_seconds=elapsed_checkpoint,
            wall_seconds=time.monotonic()-started,world_size=world,initial_lora_sha256=initial_hash,
            final_lora_sha256=change_hash,main_rows_per_update=cfg['main_rows_per_update'],auxiliary_training_rows=0,
            evaluations=[{k:v for k,v in r.items() if k!='rows'} for r in all_eval])
        if rank==0:write(out/'summary.json',summary)
        status('complete',optimizer_steps=actual_step,final_reward=summary['final_reward'],stop_reason=stop_reason)
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:
        # NCCL allocates during shutdown. A run ends holding a large caching
        # allocator, which can fail the teardown after a complete run and turn a
        # finished job into a nonzero exit.
        torch.cuda.empty_cache()
        if 'rt' in locals():
            for pool_ in (getattr(rt,'pool',None),getattr(rt,'geneval_pool',None),getattr(rt,'ocr_pool',None)):
                if pool_ is not None:pool_.close()
        if distributed():dist.destroy_process_group()

def parse():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--method',choices=['paper','endpoint'],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--group-size',type=int,default=4);p.add_argument('--groups-per-update',type=int,default=1)
    p.add_argument('--expected-trajectories-per-update',type=int,
                   help='Fail before model loading unless group_size * groups_per_update matches this global count')
    p.add_argument('--unique-prompts-per-update',action='store_true',
                   help='Sample distinct prompts within each update; each contributes exactly group_size trajectories')
    p.add_argument('--updates',type=int,default=6);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--time-exponent',type=float,default=3.);p.add_argument('--time-mode',choices=['uniform','dt'],default='uniform')
    p.add_argument('--direction-normalization',choices=['state_rms','group_rms','none'],default='state_rms')
    p.add_argument('--actor-loss',choices=['legacy_mean','fdfo_first_order'],default='legacy_mean')
    p.add_argument('--train-microbatch',type=int,default=1);p.add_argument('--noise-radius',type=float,default=.001)
    p.add_argument('--batching',choices=['prompt','rows'],default='prompt',
                   help='One prompt per batch, or a flat trajectory shard whose batches mix prompts')
    p.add_argument('--future-cap',type=int,default=4);p.add_argument('--ridge',type=float,default=.1)
    p.add_argument('--fit-scope',choices=['own','own_past','traj_window','traj_all','head','pool_head','nearest_higher','nearest_success','nearest_success_group','local'],default='own',
                   help='Rows entering the fit at q. own is the shipped estimator, past j<q plus '
                        'capped future secants; the others replace both with one joint ridge over '
                        'the endpoints before q, within --fit-width of q, or all of them. pool_head is '
                        'head over every trajectory of the prompt group (stepwise_group_fit); it needs '
                        '--batching rows --shard-mode block with a per-rank row count that is a multiple '
                        'of --group-size')
    p.add_argument('--fm-weight',type=float,default=0.,
                   help='Advantage-weighted flow matching on the scored endpoints (stepwise_advantage_fm), the '
                        'DiffusionNFT update at one optimizer step per epoch. Its rank-averaged gradient is rescaled '
                        'to this multiple of the actor gradient norm and added. 0 disables. Needs --shard-mode block')
    p.add_argument('--fm-mode',choices=['signed','positive'],default='signed',
                   help='signed keeps negative advantages (NFT); positive clamps them at 0 (rejection-sampling FT)')
    p.add_argument('--fm-per-group',type=int,default=12,help='endpoints drawn per prompt group per update')
    p.add_argument('--fm-times',type=int,default=1,help='noise-time draws per drawn endpoint')
    p.add_argument('--fm-adv-clip',type=float,default=5.,help="NFT's adv_clip_max; the weight is A / clip")
    p.add_argument('--rollout-mode',choices=['sde','ode_group'],default='sde',
                   help='sde: the shipped main SDE trajectory with stopped ODE branches. ode_group: every trajectory of a '
                        'prompt group is an independent plain ODE sample (10 forwards, 1 score); the direction of a '
                        'trained trajectory is one ridge over the other endpoints of its group (sibling contrast), so the '
                        'displacements come from the initial noises instead of injected SDE noise. Needs --batching rows '
                        '--shard-mode block --independent-roots; --fit-scope and --noise-radius are ignored')
    p.add_argument('--train-per-group',type=int,default=None,
                   help='ode_group: trajectories per group that carry actor rows (default: all of them)')
    p.add_argument('--cloud-size',type=int,default=0,
                   help='ode_group: C perturbed trajectories per main trajectory (group = M x (1 + C) rows; --group-size must be '
                        'that product). Each cloud member starts from the main initial noise perturbed by a variance-preserving '
                        'Gaussian of --cloud-strength; the main direction is the ridge over its own cloud endpoints '
                        '(stepwise_group_fit.cloud_directions). Only mains carry actor rows; --train-per-group is ignored')
    p.add_argument('--cloud-strength',type=float,default=.3,help='ode_group cloud: perturbation strength s of the initial noise')
    p.add_argument('--cloud-pool-mains',action='store_true',help='ode_group cloud: also fit on the other mains and their clouds')
    p.add_argument('--direction-mode',choices=['sibling','anchor','both'],default='sibling',
                   help='ode_group: sibling = ridge over the other endpoints of the group (prior covariance, repeated along T); '
                        'anchor = DiffusionNFT one-sample value gradient on the own ODE states, A_i (y_i - x0hat_q) per state '
                        '(stepwise_group_fit.anchor_directions); both = sibling + --anchor-mix x anchor, each at unit group rms')
    p.add_argument('--anchor-mix',type=float,default=1.,help='ode_group both: weight of the anchor family relative to the sibling family')
    p.add_argument('--state-select',choices=['all','below_max','below_cap'],default='all',
                   help='rows batching (SDE main + per-state ODE completions): train state q only if its own completion '
                        'reward is below the best completion of its trajectory (below_max) or below --state-cap; the '
                        'other states get a zero direction')
    p.add_argument('--state-cap',type=float,default=.99)
    p.add_argument('--skip-zero-rows',action='store_true',help='ode_group: do not forward rows whose direction is all-zero (loss 0); skip the update when none remain')
    p.add_argument('--train-max-reward',type=float,default=None,help='ode_group: only rows with reward below this are trained (failures only); a group gives min(M, #failures) rows')
    p.add_argument('--mixed-extra-k',type=int,default=0,help='ode_group: second ODE pass of this many extra trajectories for every group with reward contrast (adaptive K)')
    p.add_argument('--prompt-weighting',choices=['uniform','residual','residual_mixed'],default='uniform',
                   help='residual: draw prompt rows with probability floor + (1 - running success rate) (stepwise_batch.PromptResidualTable)')
    p.add_argument('--residual-prior',type=float,default=.5);p.add_argument('--residual-decay',type=float,default=.7)
    p.add_argument('--residual-floor',type=float,default=.1);p.add_argument('--residual-tag-decay',type=float,default=.95)
    p.add_argument('--tag-shares',type=str,default=None,help='geneval: stratified prompt draw, e.g. color_attr=0.35,colors=0.2,position=0.15,counting=0.2,two_object=0.1')
    p.add_argument('--fit-reward-floor',type=float,default=None,help='clip the reward used by the direction fit from below (e.g. 0: GenEval clutter and empty images level)')
    p.add_argument('--target-min',type=float,default=None,help='fit-scope nearest_higher: a target completion must score above this (e.g. 0: positive partial credit only)')
    p.add_argument('--row-normalize',action='store_true',help='rows path: unit-rms direction per (row, q) at the estimator level (one flip = one vote)')
    p.add_argument('--sibling-centre',choices=['self','mean','local'],default='self',
                   help='ode_group sibling: ridge base point at the own endpoint (self term kept) or at the leave-one-out group mean '
                        '(shared covariance direction only)')
    p.add_argument('--anchor-adv-clip',type=float,default=5.,help="ode_group anchor: NFT's adv_clip_max for the group advantages")
    p.add_argument('--anchor-scale',type=float,default=.01,help='ode_group anchor/both: constant multiplier on the directions (unclipped)')
    p.add_argument('--train-select',choices=['first','abs_adv','low','high'],default='first',
                   help='ode_group: which --train-per-group rows of a group carry actor rows: the first M, the M farthest '
                        'from the group reward mean (the rare class that carries |A| on a mostly-solved prompt), the M lowest '
                        'rewards, or the M highest (ties broken at random)')
    p.add_argument('--pool-nearest',type=int,default=None,
                   help='pool_head only: keep the K rows nearest the base point in endpoint space')
    p.add_argument('--shard-mode',choices=['interleave','block'],default='interleave',
                   help='--batching rows: interleave rows across ranks (shipped) or give each rank a '
                        'contiguous block, which keeps whole prompt groups on one rank')
    p.add_argument('--fit-head-rows',type=int,default=8,
                   help='Rows the head scope keeps: the endpoints with index below this, and no '
                        'future secant term. The last endpoints carry the smallest displacements '
                        'of the trajectory, so their secants amplify reward noise over a short '
                        'chord')
    p.add_argument('--independent-roots',action='store_true',
                   help='Give every trajectory its own initial noise instead of one root per '
                        'group. The shared root is a matched baseline that only a pooled '
                        'estimator can exploit; a per-trajectory fit pays its diversity cost '
                        'without collecting it')
    p.add_argument('--fit-width',type=int,default=3,
                   help='Half width of the traj_window row set, in steps. Ignored by other scopes')
    p.add_argument('--paper-noise',type=float,default=.7);p.add_argument('--paper-sampler',choices=['flow','ddim'],default='flow')
    p.add_argument('--ddim-eta',type=float,default=.9);p.add_argument('--value-substeps',type=int,default=5)
    p.add_argument('--paper-kl-convention',choices=['flow_grpo','exact'],default='flow_grpo',
                   help="Which KL Eq.(4)'s beta multiplies. Flow-GRPO divides by std_dev_t**2 "
                        "(scripts/train_sd3.py:900) while its own log density uses "
                        "(std_dev_t*sqrt(-dt))**2, so its KL is the exact one times dt. beta=0.01 "
                        "was tuned against THAT term, so 'flow_grpo' reproduces the regulariser "
                        "they actually ran; 'exact' is the true Gaussian KL and is up to 25x "
                        "stronger at the same beta. Only the flow sampler is affected: the DDIM "
                        "schedule's sigma_t IS the transition std, so both agree there")
    p.add_argument('--paper-max-btau',type=float,default=float('inf'),
                   help='Drop transitions whose |b|/tau exceeds this. Such a step is '
                        'numerically a deterministic edge: it contributes (b/tau)^2 of '
                        'gradient variance while its reward response scales with tau. '
                        '20 removes DDIM q8 (39.75) on the SD3.5 grid and nothing else; '
                        'every other step of either schedule is below 1.8. --method paper only')
    p.add_argument('--paper-timestep-fraction',type=float,default=.99,
                   help="Flow-GRPO's config.train.timestep_fraction. Only the first "
                        "int(T*fraction) transitions get a policy-gradient term; the "
                        "shipped 0.99 drops the last one at T=10. --method paper only")
    p.add_argument('--clip-range',type=float,default=1e-4);p.add_argument('--reference-beta',type=float,default=0.)
    p.add_argument('--reference-time-weight',choices=['late5','uniform','early','power','flow_grpo'],default='late5',
                   help='Time weight of the endpoint reference penalty: (1-t)**5 (historical), uniform, '
                        'or early (1 for transitions starting above --reference-time-cut, else 0)')
    p.add_argument('--reference-time-exponent',type=float,default=4.,
                   help="exponent k of the 'power' reference time weight t**k")
    p.add_argument('--reference-time-cut',type=float,default=.74,
                   help="sigma above which 'early' applies the reference penalty")
    p.add_argument('--policy-time-cut',type=float,default=None,
                   help='If set, the policy loss is applied only to transitions starting at sigma <= this')
    p.add_argument('--policy-time-min',type=float,default=None,
                   help='If set, the policy loss is applied only to transitions starting at sigma >= this (early steps only)')
    p.add_argument('--lora-rank',type=int,default=32);p.add_argument('--lora-alpha',type=int,default=64)
    p.add_argument('--ema-decay',type=float,default=.9);p.add_argument('--eval-every',type=int,default=2)
    p.add_argument('--ema-restart-every',type=int,default=0,
                   help='Every N optimizer steps, load the EMA into the live LoRA after the EMA update (centre restart; '
                        'AdamW moments kept). 0 disables')
    p.add_argument('--ema-update-interval',type=int,default=1,
                   help='Step the EMA only every N optimizer steps, as Flow-GRPO does '
                        '(its ema.py; the PickScore config uses 8). The same decay stepped every '
                        'N updates looks back N times further, so this and --ema-decay have to be '
                        'set together. 1 is this trainer\'s own behaviour')
    p.add_argument('--ema-warmup',action='store_true',
                   help='Ramp the decay as Flow-GRPO does, min((1+n)/(10+n), --ema-decay), with n '
                        'the optimizer steps already taken. It reaches the nominal decay only at '
                        'n=80, so over a 200-update run it governs most of the early curve')
    p.add_argument('--save-every',type=int,default=6);p.add_argument('--eval-steps',type=int,default=40)
    p.add_argument('--eval-grid-mix',type=str,default=None,help="eval only: integrate on a HYBRID grid 'HEAD/TAIL@SPLIT' "
                   "(e.g. 40/10@0.28) - nodes above SPLIT from the HEAD-step grid, at or below it from the TAIL-step grid. "
                   "localises Delta_grid in sigma (stepwise_runtime.sigma_nodes); --eval-steps is ignored when set")
    p.add_argument('--train-state-stride',type=int,default=1,
                   help='ode_group: train only the states q = phase (mod K) per update, phase cycling with the step; '
                        '1 = every state. With --main-steps 40 and K 4 the rollout sees the 40-step grid while the backward keeps '
                        'the 10-state cost, and all 40 sigmas are trained over any 4 consecutive updates.')
    p.add_argument('--main-steps',type=int,default=10,
                   help='rollout / training step count of the main trajectories (train on the 40-step grid the '
                        'evaluation and DiffusionNFT use; the SD3.5 shifted schedule makes the 10-step sigmas a non-subset of the '
                        '40-step ones, so every eval query is off the training grid). Rows per trajectory scale with it; the '
                        'sibling ode_group path is generic in T, fit_head_rows / future_cap are in step units on the other paths.')
    p.add_argument('--save-lora-every',type=int,default=0,
                   help='Also write a LoRA-only checkpoint this often. 72 MB against a full '
                        'checkpoint\'s 287 MB, so every update is affordable, and it makes SWA '
                        'windows available at update granularity. Not resumable: --resume needs '
                        'the optimizer, which only --save-every checkpoints carry')
    p.add_argument('--eval-first',action='store_true');p.add_argument('--save-first',action='store_true')
    p.add_argument('--eval-seeds',type=int,default=2);p.add_argument('--eval-seed',type=int,default=0)
    p.add_argument('--runtime-dtype',choices=['float32','bfloat16','float16'],default='bfloat16')
    p.add_argument('--scorer-dtype',choices=['float32','bfloat16','float16'],default='bfloat16')
    p.add_argument('--reward-vae-dtype',choices=['inherit','float32','bfloat16'],default='inherit',
                   help='VAE precision of the TRAINING PickScore rewards only (evaluation keeps the runtime VAE). '
                        'inherit = runtime dtype, the historical behaviour. The BF16 decode adds a '
                        'noise floor to reward differences')
    p.add_argument('--rollout-batching',choices=['serial','grouped'],default='serial',
                   help='Batch the stopped ODE suffixes that share a time node; same samples, fewer launches')
    p.add_argument('--rollout-chunk',type=int,default=0,help='Cap on samples per grouped suffix forward, 0 for none')
    p.add_argument('--score-chunk',type=int,default=1,help='VAE decode and scoring batch size')
    p.add_argument('--no-gradient-checkpointing',action='store_true',
                   help='Keep activations instead of recomputing them; needs spare memory')
    p.add_argument('--eval-chunk',type=int,default=16,help='Evaluation rows generated per transformer call')
    p.add_argument('--reward-normalization',choices=REWARD_NORMALIZATIONS,default='none',
                   help='How endpoint rewards are baselined and scaled per prompt group before the fit. '
                        'gain_running_std (head8 production): gains centred on one group x all-step mean, '
                        'divided by a running std. gain_group_std: same centring, that group\'s own std. '
                        'gain_step_mean: centred per step. gain_step_whiten: per step, and the scale shaped '
                        'by a running per-step std profile. gain_step_loo: leave-one-out per step (= '
                        'gain_step_mean x N/(N-1)). gain_step_shared: group mean plus the update-wide '
                        'per-step shape (rows batching). gain_curvature: subtract kappa_g*|y_{k+1}-y_k|^2, one '
                        'least-squares curvature per group (the second-order Taylor term the regression cannot '
                        'represent). gain_uncentered: no centring, running std. '
                        'reward_zscore: (r - mean) / std of the group\'s final rewards. See '
                        'rvge/stepwise_normalization.py')
    p.add_argument('--reward-scale',default=None,
                   help='With --reward-normalization none: divide every endpoint reward by this constant (e.g. 26 for '
                        'PickScore) or by batch_std, the std of the update\'s gathered final rewards. No baseline and no '
                        'group statistic is used, so --group-size 1 is legitimate')
    p.add_argument('--adam-eps',type=float,default=1e-8,
                   help='AdamW eps. With un-normalized rewards (/26) per-parameter gradients are tiny and LoRA A grads '
                        'start orders smaller, so 1e-8 damps them; a tiny eps (1e-15) keeps Adam reward-scale invariant')
    p.add_argument('--prepare-local-prompts',action='store_true',
                   help='Rows batching: encode only this rank\'s prompts each update (useful for hundreds of prompts per '
                        'update). Off by default, which keeps the original encoding batches')
    p.add_argument('--grad-clip',type=float,default=1.,
                   help='Gradient norm threshold. Gain normalization multiplies the raw gradient '
                        'magnitude, so it has to be retuned alongside --reward-normalization: a threshold '
                        'that clips every update degenerates the update to a unit-norm direction '
                        'times the learning rate')
    p.add_argument('--gain-std-decay',type=float,default=.9,
                   help='Decay of the running gain standard deviation used by gain_running_std')
    p.add_argument('--time-weight-mode',choices=['power','inverse_rms'],default='power',
                   help='Time weights from t**time_exponent, or from the measured per-time direction rms')
    p.add_argument('--trainable-dtype',choices=['inherit','float32','bfloat16','float16'],default='inherit',
                   help='Precision of the LoRA parameters and therefore of the AdamW moments')
    p.add_argument('--direction-index',choices=['current','next'],default='current',
                   help="Value gradient paired with row q: its own state, or the state its velocity produces")
    p.add_argument('--sde-noise-mode',choices=['state_rms','constant'],default='state_rms')
    p.add_argument('--sde-schedule',choices=[s for s in SDE_SCHEDULES if s!='state_rms'],default='flow',
                   help='Shape of the main-trajectory noise over physical time. flow is the shipped '
                        'Flow-GRPO tau=eta*sqrt(t/(1-t))*sqrt(dt); flow_annealed multiplies it by '
                        't**--sde-noise-exponent to clean up the tail; ddim is the Stepwise-CA '
                        'Eq. (8)/(9) schedule tau=eta*s*sqrt(1-t), which is zero at both ends of '
                        'the grid and therefore leaves the first and last endpoint secants empty. '
                        'eta is --noise-radius and has to be rescaled per schedule: the mean of tau '
                        'over the ten nodes is 0.572*eta for flow, 0.462*eta at exponent 1 and '
                        '0.189*eta for ddim')
    p.add_argument('--sde-noise-exponent',type=float,default=0.,
                   help='Annealing exponent of --sde-schedule flow_annealed. 0 reproduces flow bitwise')
    p.add_argument('--geneval-train-reward',choices=['score','strict','mix','soft','soft_mix','soft1'],default='score',
                   help='GenEval training reward: score = partial credit (DiffusionNFT / Flow-GRPO), strict = exact '
                        'judgement, mix = (score + strict) / 2, soft = same detector + CLIP with sigmoid / probability '
                        'clauses (rvge/stepwise_geneval.py:soft_image_score), soft_mix = (score + soft) / 2. '
                        'Evaluation records always use the discrete GenEval judgement')
    p.add_argument('--source-shares',type=str,default=None,help='multi: share of each update\'s groups per prompt source, e.g. pickapic=0.5,geneval=0.25,ocr=0.25 (fixed counts, stepwise_batch.select_prompt_ids_by_source)')
    p.add_argument('--quality-lambda',type=float,default=10.,help='multi: weight of the quality term q = PickScore/26 + ClipScore + HPSv2.1 in the GenEval / OCR groups')
    p.add_argument('--quality-mode',choices=['linear','floor','group_floor','task_floor'],default='linear',help='multi: quality term shape in rule groups - linear (q - ref), floor min(q - ref, 0) with a fixed per-source ref, group_floor min(q - mean q of the group, 0), or task_floor = the MIRROR for a quality stage: rule rows get pickapic_scale*q + task_lambda*min(task - mean task of the group, 0)')
    p.add_argument('--task-lambda',type=float,default=1.,help='multi + --quality-mode task_floor: weight of the one-sided TASK floor on rule rows. Separate from --quality-lambda because the scales differ')
    p.add_argument('--quality-ref',type=str,default=None,help='multi: per-source q reference, e.g. geneval=1.25,ocr=1.20 (base mean q of the source + margin)')
    p.add_argument('--pickapic-scale',type=float,default=10.,help='multi: Pick-a-Pic group reward = scale * q (sets its direction norm against the rule groups)')
    p.add_argument('--no-quality-swap',action='store_true',help='multi: keep the 9.6 GB of quality scorers resident on the GPU instead of moving them in only while scoring')
    p.add_argument('--task-exact-bonus',type=float,default=0.,help='multi: bonus added to rule rows with task >= 0.99 (plateau contrast; stepwise_quality.compose_multi_rewards)')
    p.add_argument('--quality-weights',type=str,default=None,help="multi: per-component weights of q, e.g. 'pickscore=1.3,clipscore=1.3,hpsv2=0.4' (default 1/1/1 = NFT's equal weighting); keep the sum at 3 so the pickapic reward scale is unchanged")
    p.add_argument('--quality-margin',type=str,default='0',help="multi floor modes: the floor sits this far below the reference (group mean q or fixed ref). A scalar applies to both rule sources; 'ocr=0.03,geneval=0' sets it per source")
    p.add_argument('--reward',choices=['pickscore','ocr','geneval'],default='pickscore',
                   help='pickscore is the raw PickScore on float images. ocr is the Flow-GRPO '
                        'OCR reward (PaddleOCR, 1 - edit distance / target length, target = the '
                        'quoted span of the prompt) on uint8 images, scored in a CPU process pool')
    p.add_argument('--ocr-workers',type=int,default=16,
                   help='PaddleOCR worker processes behind each rank\'s OCR server')
    p.add_argument('--ocr-backend',choices=['cpu','gpu'],default='cpu',
                   help='cpu is the Flow-GRPO/DiffusionNFT setting. Its text detector is memory '
                        'bound on CPU, while one update '
                        'scores 11 endpoints per trajectory; gpu runs the same PaddleOCR models on '
                        'the rank\'s own card from the paddle_gpu venv')
    p.add_argument('--ocr-gpu-mem',type=int,default=3000,help='PaddleOCR gpu_mem (MB) per GPU worker')
    p.add_argument('--mix-pickscore-lambda',type=float,default=0.,
                   help='OCR reward only: training reward = OCR + lambda * (PickScore - ref) / 26, scored by the '
                        'fp32 PickScore scorer on the same decoded images; evaluation stays OCR-only. Per-step ocr_mean / '
                        'pickscore_mean of the training endpoints are logged as train_ocr_mean / train_pickscore_mean')
    p.add_argument('--mix-pickscore-ref',type=float,default=20.,help='PickScore reference subtracted in the mixed reward (base ~20)')
    p.add_argument('--mix-pickscore-clamp',action='store_true',
                   help='quality floor: the PickScore term is lambda * min(PickScore - ref, 0) / 26, zero above ref')
    p.add_argument('--geneval-batch',type=int,default=4,
                   help='Images per detector batch in each rank\'s GenEval server. Detector memory grows with the batch, '
                        'so raise it when the card has the room')
    p.add_argument('--eval-prompts-file',type=Path)
    p.add_argument('--eval-prompt-limit',type=int)
    p.add_argument('--max-hours',type=float,default=16.);p.add_argument('--resume',type=Path)
    p.add_argument('--eval-policy',choices=['current','ema','both'],default='current',
                   help='Weights the evaluation runs on. The EMA is maintained either way; '
                        'in a diffusion dominated plateau the live weights sit on a random '
                        'walk whose centre the average recovers')
    p.add_argument('--eval-cfg',type=float,action='append',
                   help='Classifier-free guidance scale for evaluation, repeatable for a sweep '
                        '(default 1.0, which is what training samples at). Training itself is '
                        'unguided: the loss would need re-deriving under guidance, so this is a '
                        'transfer check of the learned policy, not a change of objective')
    p.add_argument('--eval-only',action='store_true',
                   help='Evaluate the starting policy and exit without taking an optimizer step')
    p.add_argument('--lr-schedule',choices=['constant','cosine','linear'],default='constant',
                   help='Learning-rate schedule from --lr down to --lr-final over the updates '
                        'after --lr-decay-start')
    p.add_argument('--lr-final',type=float,default=0.,help='Terminal learning rate of the schedule')
    p.add_argument('--lr-decay-start',type=int,default=0,help='Update at which the schedule begins')
    p.add_argument('--resume-weights-only',action='store_true',
                   help='resume the LoRA and EMA from the checkpoint but start the optimizer fresh; use when the '
                        'resumed run changes the gradient scale (estimator / reward normalization / reward scale)')
    p.add_argument('--allow-resume-lr-change',action='store_true',
                   help='Permit a resumed run to use a different learning rate than the checkpoint '
                        'carried. lr_at(step) overwrites param_groups every update anyway, so the '
                        'refusal without this flag is a guard against changing the step size by '
                        'accident, not an algorithmic constraint. AdamW\'s second moment is inherited '
                        'and adapts on a 1/(1-beta2) = 1000 update timescale, so the new rate acts on '
                        'the old gradient scale for the whole of a short continuation. The change and '
                        'the checkpoint rate are recorded in resume_state.json')
    p.add_argument('--gain-std-fixed',type=float,
                   help='Freeze the gain_running_std scale at this value instead of tracking it. '
                        'A running estimate that follows a decaying signal keeps the step size '
                        'constant while the signal shrinks, which turns signal decay into noise')
    p.add_argument('--gain-std-fixed-start',type=int,default=0,
                   help='First update that uses --gain-std-fixed; earlier updates use the running estimate. '
                        'A fixed scale from step 0 divides the early gains by more and so strengthens any KL '
                        'term relative to them')
    return p.parse_args()
if __name__=='__main__':train(parse())
