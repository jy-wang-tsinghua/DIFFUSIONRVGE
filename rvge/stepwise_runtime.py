"""BF16-loaded SD3.5 runtime shared by both training methods.

The base model, VAE, and PickScore scorer load in BF16, while the cached prompt
embeddings, velocity outputs, and reward scalars are promoted back to FP32 for
the sensitive rollout / optimization arithmetic.
"""
import gc
import os
from collections import OrderedDict
from pathlib import Path
import sys
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from rvge.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

# Read-only, previously validated floating-image PickScore preprocessing.
from rvge.pickscore import DifferentiableRawPickScore

TARGETS=['attn.add_k_proj','attn.add_q_proj','attn.add_v_proj','attn.to_add_out',
         'attn.to_k','attn.to_out.0','attn.to_q','attn.to_v']
# Prompt sources of the multi reward, in id order (bank['src'] holds the index).
SOURCES=('pickapic','geneval','ocr')


class TrainingRuntime:
    def __init__(
        self,
        seed=0,
        rank=32,
        alpha=64,
        device='cuda:0',
        steps=10,
        runtime_dtype=torch.bfloat16,
        scorer_dtype=torch.bfloat16,
        trainable_dtype=None,
        score_chunk=1,
        cache_limit=1024,
        gradient_checkpointing=True,
        reward='pickscore',
        ocr_workers=16,
        ocr_backend='cpu',
        ocr_gpu_mem=3000,
        reward_metadata=None,
        geneval_batch=4,
        geneval_train_reward='score',
        reward_vae_dtype=None,
        reward_decode_chunk=None,
        mix_pickscore_lambda=0.,
        mix_pickscore_ref=20.,
        mix_pickscore_clamp=False,
        prompt_sources=None,
        quality_lambda=10.,
        quality_mode='linear',
        quality_ref=None,
        pickapic_scale=10.,
        quality_weights=None,
        quality_swap=True,
    ):
        # 'none' loads no scorer at all, for pure generation (checkpoint renders).
        if reward not in ('pickscore','ocr','geneval','none','multi'):raise ValueError(f'unknown reward {reward!r}')
        self.reward=reward
        # reward='multi' (NFT Table 1 joint model): every row is scored by
        # the three model-based rewards q = PickScore/26 + ClipScore + HPSv2.1 (stepwise_quality) and, by the SOURCE of its
        # prompt, by the rule-based task reward (GenEval detector / PaddleOCR). Composite training reward per row:
        #   pickapic   r = pickapic_scale * q
        #   geneval    r = GenEval score + quality_lambda * phi(q - quality_ref['geneval'])
        #   ocr        r = OCR          + quality_lambda * phi(q - quality_ref['ocr'])
        # phi = identity ('linear': the quality term keeps taking the direction once the task saturates) or
        # min(., 0) ('floor': quality only pushes rows below the reference, the task owns the rest).
        # Evaluation reports the task metric of rule rows and the raw PickScore of Pick-a-Pic rows, per source.
        self.sources=dict(prompt_sources or {})
        self.quality=None;self.geneval_pool=None;self.ocr_pool=None
        self.quality_lambda=float(quality_lambda);self.quality_mode=quality_mode;self.pickapic_scale=float(pickapic_scale)
        self.quality_ref={k:float(v) for k,v in (quality_ref or {}).items()}
        self.quality_weights={k:float(v) for k,v in (quality_weights or {}).items()} or None
        self.multi_decode_chunk=11
        if reward=='multi':
            # 'group_floor': min(q - mean q of the row's own group, 0) - the floor follows the model's current
            # quality on that prompt, so it keeps firing on degenerate rows after the Pick-a-Pic groups have lifted every
            # source above any fixed reference. Composed in the trainer (stepwise_quality.compose_multi_rewards), where
            # the groups are known; score_multi returns the linear form and the components.
            if quality_mode not in ('linear','floor','group_floor','task_floor'):raise ValueError(f'unknown quality mode {quality_mode!r}')
            if not self.sources:raise ValueError('multi reward needs the prompt -> source map of the panel')
            bad=sorted(set(self.sources.values())-set(SOURCES))
            if bad:raise ValueError(f'unknown prompt sources {bad}; expected {SOURCES}')
            for src in ('geneval','ocr'):
                if quality_mode=='floor' and src in self.sources.values() and src not in self.quality_ref:
                    raise ValueError(f'quality mode floor needs quality_ref[{src!r}] (base mean q of that source + margin)')
        # OCR with a PickScore term: training reward = OCR + lambda * (PickScore - ref) / 26. A quality term in the ODE-group
        # covariance direction credits legible text inside a scene rather than on a flat template. Evaluation stays OCR-only (evaluate_rows uses the
        # pool path), so the test-set ruler does not move. NFT Table 1 trains OCR jointly with PickScore/CLIP/HPS.
        if mix_pickscore_lambda and reward!='ocr':raise ValueError('mix_pickscore_lambda is implemented for the OCR reward only')
        self.mix_lambda=float(mix_pickscore_lambda);self.mix_ref=float(mix_pickscore_ref);self.last_mix=None
        # clamp: quality floor - only rows BELOW ref get the (negative) quality term, so above the
        # floor the covariance direction is pure OCR; the floor holds the PickScore gate instead of trading OCR for it.
        self.mix_clamp=bool(mix_pickscore_clamp)
        self.device=torch.device(device)
        self.pool=None
        if reward=='ocr':
            # Started before the model loads: a GPU server has to find its memory before
            # PyTorch's caching allocator has reserved the card.
            from rvge.stepwise_ocr import OcrPool
            self.pool=OcrPool(workers=ocr_workers,backend=ocr_backend,gpu_id=self.device.index or 0,
                             gpu_mem=ocr_gpu_mem)
        elif reward=='geneval':
            # Same placement rule: the detector reserves its activation memory first.
            from rvge.stepwise_geneval import GenevalPool
            if not reward_metadata:raise ValueError('geneval reward needs the panel metadata map')
            self.pool=GenevalPool(reward_metadata,gpu_id=self.device.index or 0,batch=geneval_batch,
                                  train_reward=geneval_train_reward)
        elif reward=='multi':
            # Both servers on every rank (any group may land on any rank; the block shard keeps a group whole): the
            # detector 5.5 GB + PaddleOCR 3 GB reserve first, the quality scorers are swapped in only while scoring.
            present=set(self.sources.values())
            if 'geneval' in present:
                from rvge.stepwise_geneval import GenevalPool
                if not reward_metadata:raise ValueError('multi reward with GenEval rows needs the panel metadata map')
                self.geneval_pool=GenevalPool(reward_metadata,gpu_id=self.device.index or 0,batch=geneval_batch,
                                              train_reward=geneval_train_reward)
            if 'ocr' in present:
                from rvge.stepwise_ocr import OcrPool
                self.ocr_pool=OcrPool(workers=ocr_workers,backend=ocr_backend,gpu_id=self.device.index or 0,gpu_mem=ocr_gpu_mem)
        self.last_eval_details=None
        self.runtime_dtype=runtime_dtype
        self.scorer_dtype=scorer_dtype
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        pipe=StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium',
                    torch_dtype=runtime_dtype,local_files_only=True)
        self.encoders=[pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3]
        self.tokenizers=[pipe.tokenizer,pipe.tokenizer_2,pipe.tokenizer_3]
        for enc in self.encoders:enc.requires_grad_(False).eval()
        pipe.text_encoder=pipe.text_encoder_2=pipe.text_encoder_3=None
        self.model=pipe.transformer.to(self.device,dtype=runtime_dtype).requires_grad_(False).eval()
        # Checkpointing trades recomputation for memory. Peak use is dominated by the
        # frozen weights and the optimizer, so spare memory buys back the recompute.
        if gradient_checkpointing:self.model.enable_gradient_checkpointing()
        torch.manual_seed(seed)
        self.model.add_adapter(LoraConfig(r=rank,lora_alpha=alpha,init_lora_weights='gaussian',target_modules=TARGETS))
        # The adapter inherits the frozen base dtype, which also sets the precision of
        # AdamW's moments. An optimizer step near lr/|w| = 2**-8 is one BF16 ulp, so
        # promoting only the trainable tensors keeps small steps representable.
        if trainable_dtype is not None:
            for p in self.model.parameters():
                if p.requires_grad:p.data=p.data.to(trainable_dtype)
        self.params=[p for p in self.model.parameters() if p.requires_grad]
        self.trainable_dtype=self.params[0].dtype
        self.vae=pipe.vae.to(self.device,dtype=runtime_dtype).requires_grad_(False).eval()
        # Training PickScore rewards (score / score_rows) may decode with their own VAE precision; evaluation and
        # decode() keep self.vae, so the readout ruler does not move. The BF16 decode adds a chord-independent
        # floor to reward differences, which the chord-normalized regression rows then amplify by 1/|dy|.
        self.reward_vae,self.reward_vae_dtype=self.vae,runtime_dtype
        # An FP32 decode needs about twice the BF16 activation memory, so it may run in smaller pieces. None keeps one decode per scoring chunk, bit for bit the historical path.
        self.reward_decode_chunk=None
        if reward_vae_dtype is not None and reward_vae_dtype!=runtime_dtype:
            self.reward_decode_chunk=reward_decode_chunk or 11
            if reward!='pickscore':raise ValueError('reward_vae_dtype only applies to the PickScore reward')
            from diffusers import AutoencoderKL
            self.reward_vae=AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-3.5-medium',subfolder='vae',
                torch_dtype=reward_vae_dtype,local_files_only=True).to(self.device).requires_grad_(False).eval()
            self.reward_vae_dtype=reward_vae_dtype
        # OCR needs no GPU scorer: skipping PickScore frees its ~4 GB of fp32 CLIP-H.
        self.scorer=(DifferentiableRawPickScore(device=self.device,dtype=scorer_dtype,local_files_only=True)
                     if reward in('pickscore','multi') or self.mix_lambda else None)
        if reward=='multi':
            from rvge.stepwise_quality import QualityScorers
            self.quality=QualityScorers(self.device,scorer_dtype,pickscore=self.scorer,swap=quality_swap)
        self.scheduler=pipe.scheduler
        self.scheduler.set_timesteps(steps,device=self.device)
        self.times=[float(t) for t in self.scheduler.sigmas]
        self.shape=(int(self.model.config.in_channels),64,64)
        self.cache=OrderedDict()
        self.cache_limit=cache_limit
        self.score_chunk=score_chunk
        self.nfe=0;self.forward_samples=0;self.score_samples=0
        self.encoded_prompts=0

    @torch.no_grad()
    def prepare(self,prompts):
        """Encode every missing prompt in one pass and mark the rest as recently used.

Call this once per set of prompts. Calling it per prompt moves the text
encoders, T5-XXL included, across PCIe on every call.
"""
        for p in prompts:
            if p in self.cache:self.cache.move_to_end(p)
        missing=list(dict.fromkeys(p for p in prompts if p not in self.cache))
        if missing:
            # text_encoder_device='cpu' (side renders next to a running 8-GPU job): the 9.5 GB T5 pass stays
            # off the card; the cached embeddings are CPU fp32 either way.
            for enc in self.encoders:enc.to(getattr(self,'text_encoder_device',None) or os.environ.get('STEPWISE_TEXT_ENCODER_DEVICE') or self.device)
            for start in range(0,len(missing),4):
                batch=missing[start:start+4]
                emb,pool=encode_prompt(self.encoders,self.tokenizers,batch,128)
                if self.quality is not None:
                    # All three text towers at once; the PickScore feature is also kept in the runtime cache slot.
                    self.quality.prepare(batch)
                    text=torch.cat([self.quality.text[p]['pickscore'] for p in batch]).float()
                else:text=self.scorer.encode_text(batch).float() if self.scorer is not None else None
                for i,p in enumerate(batch):
                    self.cache[p]=(emb[i:i+1].float().cpu(),pool[i:i+1].float().cpu(),
                                   text[i:i+1].cpu() if text is not None else None)
                    self.encoded_prompts+=1
            for enc in self.encoders:enc.to('cpu')
            gc.collect();torch.cuda.empty_cache()
        # Trimming from the front cannot drop what this call just touched, provided
        # the limit exceeds the largest prompt set handed to one call.
        while len(self.cache)>self.cache_limit:self.cache.popitem(last=False)

    def conditioning(self,prompts):
        """Stack one conditioning row per sample so a batch may mix prompts."""
        emb=torch.cat([self.cache[p][0] for p in prompts]).to(self.device,dtype=self.runtime_dtype)
        pool=torch.cat([self.cache[p][1] for p in prompts]).to(self.device,dtype=self.runtime_dtype)
        return emb,pool

    def velocity(self,x,t,prompt):
        self.nfe+=1;self.forward_samples+=len(x)
        emb,pool,_=self.cache[prompt]
        return self.model(hidden_states=x.to(dtype=self.runtime_dtype),
                timestep=torch.full((len(x),),float(t)*1000,device=self.device,dtype=torch.float32),
                encoder_hidden_states=emb.to(self.device,dtype=self.runtime_dtype).expand(len(x),-1,-1),
                pooled_projections=pool.to(self.device,dtype=self.runtime_dtype).expand(len(x),-1),return_dict=False)[0].float()

    def velocity_rows(self,x,t,prompts):
        """Velocity for a batch whose rows may carry different prompts.

Nothing in the estimator or the loss couples trajectories, so batching is free
to cross prompt boundaries; only the conditioning has to be gathered per row.
"""
        if len(prompts)!=len(x):
            raise ValueError('one prompt per row is required')
        self.nfe+=1;self.forward_samples+=len(x)
        emb,pool=self.conditioning(prompts)
        return self.model(hidden_states=x.to(dtype=self.runtime_dtype),
                timestep=torch.full((len(x),),float(t)*1000,device=self.device,dtype=torch.float32),
                encoder_hidden_states=emb,pooled_projections=pool,return_dict=False)[0].float()

    @torch.no_grad()
    def decode(self,y,chunk=None):
        """[0,1] float images of latents, decoded `chunk` at a time exactly as the scorers see them."""
        out=[]
        for yi in y.split(chunk or self.score_chunk):
            z=(yi/self.vae.config.scaling_factor+self.vae.config.shift_factor).to(self.runtime_dtype)
            out.append((self.vae.decode(z,return_dict=False)[0]/2+.5).clamp(0,1))
        return torch.cat(out)

    @torch.no_grad()
    def submit_external(self,y,prompts,chunk=None,evaluation=False):
        from rvge.stepwise_ocr import to_uint8_nhwc
        images=np.concatenate([to_uint8_nhwc(self.decode(yi,chunk)) for yi in y.split(chunk or self.score_chunk)])
        # The VAE decode just filled PyTorch's cache; hand it back so a GPU OCR worker that
        # outgrows its reservation can still allocate.
        if self.pool.backend=='gpu':torch.cuda.empty_cache()
        return self.pool.submit(images,list(prompts),evaluation=evaluation)

    def reward_decoder(self,vae=None):
        """(vae, dtype) that training rewards decode with; pass self.vae to force the evaluation decoder."""
        if vae is None:return self.reward_vae,self.reward_vae_dtype
        return vae,next(vae.parameters()).dtype

    def reward_images(self,vae,z):
        """Decode one scoring chunk; an overridden reward VAE decodes in reward_decode_chunk pieces."""
        if self.reward_decode_chunk is None or vae is not self.reward_vae or vae is self.vae:
            return (vae.decode(z,return_dict=False)[0]/2+.5).clamp(0,1)
        return torch.cat([(vae.decode(piece,return_dict=False)[0]/2+.5).clamp(0,1)
                          for piece in z.split(self.reward_decode_chunk)])

    def source_of(self,prompt):
        try:return self.sources[prompt]
        except KeyError as e:raise KeyError(f'prompt has no source in the multi panel: {prompt!r}') from e

    @torch.no_grad()
    def score_multi(self,y,prompts,chunk=None,evaluation=False):
        """Composite multi-reward of latents `y`: one decode per chunk feeds the float image to the three quality
scorers and the uint8 image to the task server of the row's source. Returns the composite reward [n]; the
components are left in self.last_mix (source ids, task, pickscore, clipscore, hpsv2, q) and, for evaluation,
self.last_eval_details holds one record per row (task fields of the GenEval / OCR servers included)."""
        from rvge.stepwise_ocr import to_uint8_nhwc
        from rvge.stepwise_quality import QUALITY_KEYS, composite_quality
        src=[self.source_of(p) for p in prompts]
        step_=chunk or self.score_chunk
        parts={};u8=[]   # keys: QUALITY_KEYS + 'aes' when the aesthetic instrument is on (stepwise_quality)
        # The VAE decode dominates this phase's memory, so it runs in pieces of multi_decode_chunk inside each scoring chunk.
        with self.quality.scoring():
            for lo in range(0,len(y),step_):
                img=self.decode(y[lo:lo+step_],self.multi_decode_chunk)
                qd=self.quality.score(img.float(),prompts[lo:lo+step_],aesthetic=evaluation)
                for k,v in qd.items():parts.setdefault(k,[]).append(v)
                u8.append(to_uint8_nhwc(img))
        u8=np.concatenate(u8)
        task=torch.zeros(len(y),dtype=torch.float32);details=[dict(source=s) for s in src];handles=[]
        for name,pool in(('geneval',self.geneval_pool),('ocr',self.ocr_pool)):
            idx=[i for i,s_ in enumerate(src) if s_==name]
            if not idx:continue
            if pool is None:raise RuntimeError(f'{name} rows but no {name} server in this runtime')
            if pool.backend=='gpu':torch.cuda.empty_cache()
            handles.append((name,pool,idx,pool.submit(u8[idx],[prompts[i] for i in idx],evaluation=evaluation)))
        for name,pool,idx,h in handles:
            if evaluation and hasattr(pool,'result_details'):
                values,rows=pool.result_details(h)
                for i,row in zip(idx,rows):details[i].update(row)
            else:values=pool.result(h)
            task[idx]=torch.tensor(values,dtype=torch.float32)
        pick,clip,hps=(torch.cat(parts[k]) for k in QUALITY_KEYS)
        aes=torch.cat(parts['aes']) if 'aes' in parts else None
        q=composite_quality(pick,clip,hps,self.quality_weights)
        sid=torch.tensor([SOURCES.index(s_) for s_ in src])
        from rvge.stepwise_quality import compose_multi_rewards
        r=compose_multi_rewards(sid,task,q,self.pickapic_scale,self.quality_lambda,
                                'linear' if self.quality_mode in ('group_floor','task_floor') else self.quality_mode,self.quality_ref)
        pa=sid==0
        self.last_mix=dict(source=sid,task=task,pickscore=pick,clipscore=clip,hpsv2=hps,q=q,n=len(y),aes=aes)
        for i,d in enumerate(details):
            d.update(pickscore=float(pick[i]),clipscore=float(clip[i]),hpsv2=float(hps[i]),q=float(q[i]))
            if aes is not None:d['aes']=float(aes[i])
            if src[i]=='ocr':d['ocr']=float(task[i])
        self.last_eval_details=details if evaluation else None
        self.score_samples+=len(y)
        # Evaluation rows read the task metric of rule rows and the raw PickScore of Pick-a-Pic rows.
        if evaluation:return torch.where(pa,pick,task)
        return r

    @torch.no_grad()
    def score_rows(self,y,prompts,chunk=None,vae=None):
        if len(prompts)!=len(y):
            raise ValueError('one prompt per row is required')
        if self.reward=='multi':return self.score_multi(y,prompts,chunk)
        if self.pool is not None and self.mix_lambda:
            # One decode per chunk feeds both scorers: uint8 to the OCR server, the float image to fp32 PickScore.
            from rvge.stepwise_ocr import to_uint8_nhwc
            text=torch.cat([self.cache[p][2] for p in prompts]).to(self.device,dtype=self.scorer_dtype)
            u8=[];pick=[];step_=chunk or self.score_chunk
            for lo in range(0,len(y),step_):
                img=self.decode(y[lo:lo+step_],step_)
                u8.append(to_uint8_nhwc(img))
                pick.append(self.scorer.score_with_text_features(img.to(self.scorer_dtype),text[lo:lo+step_]).float().cpu())
            if self.pool.backend=='gpu':torch.cuda.empty_cache()
            ocr=torch.tensor(self.pool.result(self.pool.submit(np.concatenate(u8),list(prompts),evaluation=False)),dtype=torch.float32)
            pick=torch.cat(pick)
            self.last_mix=dict(ocr_mean=float(ocr.mean()),pickscore_mean=float(pick.mean()),n=len(y),ocr=ocr,pickscore=pick)
            self.score_samples+=len(y)
            term=(pick-self.mix_ref).clamp(max=0.) if self.mix_clamp else (pick-self.mix_ref)
            return ocr+self.mix_lambda*term/26.
        if self.pool is not None:
            scores=torch.tensor(self.pool.result(self.submit_external(y,prompts,chunk)),dtype=torch.float32)
            self.score_samples+=len(y)
            return scores
        text=torch.cat([self.cache[p][2] for p in prompts]).to(self.device,dtype=self.scorer_dtype)
        vae,vae_dtype=self.reward_decoder(vae)
        scores=[]
        for lo in range(0,len(y),chunk or self.score_chunk):
            hi=min(lo+(chunk or self.score_chunk),len(y))
            z=(y[lo:hi]/vae.config.scaling_factor+vae.config.shift_factor).to(vae_dtype)
            img=self.reward_images(vae,z)
            scores.append(self.scorer.score_with_text_features(img,text[lo:hi]))
        self.score_samples+=len(y)
        return torch.cat(scores).float().detach()

    def noise(self,seed,batch=1):
        gen=torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn((batch,*self.shape),generator=gen,device=self.device,dtype=torch.float32)

    @torch.no_grad()
    def score(self,y,prompt,chunk=None):
        if self.pool is not None:return self.score_rows(y,[prompt]*len(y),chunk)
        # Decode chunk trades memory for utilization; batch one leaves the VAE idle.
        scores=[]
        text=self.cache[prompt][2].to(self.device,dtype=self.scorer_dtype)
        vae,vae_dtype=self.reward_decoder()
        for yi in y.split(chunk or self.score_chunk):
            z=(yi/vae.config.scaling_factor+vae.config.shift_factor).to(vae_dtype)
            img=self.reward_images(vae,z)
            # One cached text feature per prompt; the scorer pairs them row by row.
            scores.append(self.scorer.score_with_text_features(img,text.expand(len(yi),-1)))
        self.score_samples+=len(y)
        return torch.cat(scores).float().detach()

    @torch.no_grad()
    def evaluate(self,prompt,seeds,steps=40):
        self.prepare([prompt])
        return self.evaluate_rows([(prompt,s) for s in seeds],steps,len(seeds))

    NEGATIVE_PROMPT=''

    def sigma_nodes(self,steps,mix=None):
        """Integration nodes of an evaluation: the scheduler's `steps` grid, or a HYBRID of two grids.

`mix` is 'HEAD/TAIL@SPLIT': the nodes above SPLIT come from the HEAD-step
grid and the nodes at or below it from the TAIL-step grid. It exists to localise Delta_grid in sigma without
touching the ruler: '40/10@0.28' is the ruler's head with our training grid's single final jump, '10/40@0.28' the
mirror. Under shift=3 the 10-step grid's last real step covers sigma 0.278 -> 0.0089 in one move while the 40-step
grid takes five, and glyphs are formed in exactly that range - so the two cells decompose the split additively.
"""
        def grid(n):
            self.scheduler.set_timesteps(n,device=self.device)
            return [float(t) for t in self.scheduler.sigmas]
        if not mix:
            return grid(steps)
        head,_,rest=mix.partition('/');tail,_,split=rest.partition('@')
        head,tail,split=int(head),int(tail),float(split)
        nodes=[t for t in grid(head) if t>split]+[t for t in grid(tail) if t<=split]
        if len(nodes)<3 or nodes[0]<=split or nodes[-1]!=0.:
            raise ValueError(f'--eval-grid-mix {mix!r} gives a degenerate node list {nodes[:3]}...{nodes[-2:]}')
        return nodes

    @torch.no_grad()
    def evaluate_rows(self,rows,steps=40,chunk=16,guidance=1.,grid_mix=None):
        """Deterministic ODE reward for (prompt, seed) rows, batched across prompts.

Prompts are padded to one sequence length at encode time, so rows with different
prompts stack into a single transformer call. One row per call leaves the GPU
almost idle, which dominates evaluation on a large held-out set.

guidance above 1 applies classifier-free guidance against the empty prompt. The
conditional and unconditional velocities are taken in two calls of the original
width rather than one call of twice the width, so each row sits in the same batch
shape it would have at guidance 1: BF16 reductions depend on that shape, and the
guidance 1 column has to stay comparable with guidance-free evaluation. It costs two forwards a node either way.
"""
        self.model.eval()
        nodes=self.sigma_nodes(steps,grid_mix)
        if guidance!=1.:self.prepare([self.NEGATIVE_PROMPT])
        out=[];pending=[];details=[]
        for start in range(0,len(rows),chunk):
            part=rows[start:start+chunk]
            x=torch.cat([self.noise(s) for _,s in part])
            emb=torch.cat([self.cache[p][0] for p,_ in part]).to(self.device,dtype=self.runtime_dtype)
            pool=torch.cat([self.cache[p][1] for p,_ in part]).to(self.device,dtype=self.runtime_dtype)
            if guidance!=1.:
                nemb,npool=self.cache[self.NEGATIVE_PROMPT][0],self.cache[self.NEGATIVE_PROMPT][1]
                nemb=nemb.to(self.device,dtype=self.runtime_dtype).expand(len(x),-1,-1)
                npool=npool.to(self.device,dtype=self.runtime_dtype).expand(len(x),-1)
            for t,s in zip(nodes,nodes[1:]):
                self.nfe+=1;self.forward_samples+=len(x)
                timestep=torch.full((len(x),),float(t)*1000,device=self.device,dtype=torch.float32)
                v=self.model(hidden_states=x.to(dtype=self.runtime_dtype),timestep=timestep,
                    encoder_hidden_states=emb,pooled_projections=pool,return_dict=False)[0].float()
                if guidance!=1.:
                    self.nfe+=1;self.forward_samples+=len(x)
                    vu=self.model(hidden_states=x.to(dtype=self.runtime_dtype),timestep=timestep,
                        encoder_hidden_states=nemb,pooled_projections=npool,return_dict=False)[0].float()
                    v=vu+guidance*(v-vu)
                x=x+(s-t)*v
            if self.reward=='multi':
                # Quality scorers and the task servers of the rows' sources; one swap per chunk is cheap next to the
                # 40-step generation of the chunk.
                out.extend(self.score_multi(x,[p for p,_ in part],self.score_chunk,evaluation=True).tolist())
                details.extend(self.last_eval_details)
                continue
            if self.pool is not None:
                # OCR runs on CPU while the next chunk generates; results collected at the end.
                pending.append(self.submit_external(x,[p for p,_ in part],evaluation=True))
                self.score_samples+=len(part)
                continue
            text=torch.cat([self.cache[p][2] for p,_ in part]).to(self.device,dtype=self.scorer_dtype)
            for lo in range(0,len(part),self.score_chunk):
                hi=min(lo+self.score_chunk,len(part))
                z=(x[lo:hi]/self.vae.config.scaling_factor+self.vae.config.shift_factor).to(self.runtime_dtype)
                img=(self.vae.decode(z,return_dict=False)[0]/2+.5).clamp(0,1)
                out.extend(self.scorer.score_with_text_features(img,text[lo:hi]).float().cpu().tolist())
            self.score_samples+=len(part)
        for handle in pending:
            if hasattr(self.pool,'result_details'):
                values,rows=self.pool.result_details(handle);out.extend(values);details.extend(rows)
            else:out.extend(self.pool.result(handle))
        # Per-row benchmark fields (GenEval tag / correct / strict) for the evaluation record.
        self.last_eval_details=details or None
        return out
