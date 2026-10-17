"""Controlled V3 fine-tuning of V7's existing candidate generator and selector."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src" / "data", ROOT / "src" / "models", ROOT / "src" / "losses"):
    sys.path.insert(0, str(p))
from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_v7 import Paper1ModelV7
from losses.paper1_loss_v7 import Paper1LossV7

EXP = ROOT / "results/task_aware_v1/enhancement_only_v3"
CKPT = ROOT / "checkpoints/v7_enhancement_only_v3"
CONFIG_PATH = EXP / "TUNING_CONFIGURATION.json"
BASELINE = ROOT / "checkpoints/v7/paper1_v7_best_epoch_019.pth"
V1 = ROOT / "checkpoints/v7_enhancement_only_v1/paper1_v7_enhancement_only_best_epoch_019.pth"
V2 = ROOT / "checkpoints/v7_enhancement_only_v2/paper1_v7_enhancement_only_best_epoch_019.pth"
DATA = ROOT / "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet"
METRIC_PY = ROOT / ".venv/Scripts/python.exe"
METRIC_SCRIPT = ROOT / "src/compute_v7_quality_metrics.py"
LOSS_KEYS = ("total", "reconstruction", "preservation", "consequence", "selection")
TRAINABLE_MODULES = ("candidate_generator", "selection_network")
FROZEN_MODULES = ("condition_encoder", "information_preservation", "consequence_check")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def module_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def scope_hash(model: Paper1ModelV7) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def capture_rng(generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader_generator": generator.get_state()}


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None: torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def load_model(path: Path, device: torch.device, trainable: bool = False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = Paper1ModelV7()
    model.load_state_dict(state["model_state_dict"], strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    if trainable:
        for name in TRAINABLE_MODULES:
            for p in getattr(model, name).parameters(): p.requires_grad_(True)
    model.to(device)
    return model, state


def set_modes(model: Paper1ModelV7, training: bool) -> None:
    model.train(training)
    for name in FROZEN_MODULES: getattr(model, name).eval()


def check_output(outputs: dict, inputs: torch.Tensor) -> None:
    image = outputs["selected_image"]
    if image.shape != inputs.shape or not torch.isfinite(image).all().item():
        raise FloatingPointError("Output shape mismatch or non-finite pixels")
    if image.min().item() < -1e-6 or image.max().item() > 1.0 + 1e-6:
        raise RuntimeError("Output outside [0,1]")


def finite_losses(losses: dict) -> bool:
    return all(torch.isfinite(x.detach()).all().item() for x in losses.values())


def make_loaders(config: dict, generator: torch.Generator):
    train = EUVPPairedDataset(DATA / "trainA", DATA / "trainB", ROOT / config["train_split"],
                              transform=PairedTransform(training=True))
    val = EUVPPairedDataset(DATA / "trainA", DATA / "trainB", ROOT / config["validation_split"],
                            transform=PairedTransform(training=False))
    if set(train.filenames) & set(val.filenames): raise RuntimeError("Train/validation identifiers overlap")
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available(), generator=generator)
    val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    return train, val, train_loader, val_loader


def optimizer_for(model: Paper1ModelV7, config: dict):
    return torch.optim.Adam([
        {"params": model.candidate_generator.parameters(), "lr": config["trainable_scope"]["candidate_generator"]["learning_rate"], "name": "candidate_generator"},
        {"params": model.selection_network.parameters(), "lr": config["trainable_scope"]["selection_network"]["learning_rate"], "name": "selection_network"},
    ])


def run_train_epoch(model, criterion, loader, optimizer, scaler, device, epoch, amp, config, events):
    set_modes(model, True); sums = {k: 0.0 for k in LOSS_KEYS}; norms = []; successful = 0; n = 0
    retries_start = len(events)
    for bi, batch in enumerate(loader, 1):
        x=batch["underwater"].to(device,non_blocking=True); y=batch["reference"].to(device,non_blocking=True)
        attempt=0
        while True:
            attempt+=1; optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                out=model(x); check_output(out,x); losses=criterion(out,y,x)
            loss_ok=finite_losses(losses)
            if loss_ok:
                scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
                params=[p for p in model.parameters() if p.requires_grad]
                grad_ok=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
                nonzero=any(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in params)
            else: grad_ok=nonzero=False
            if loss_ok and grad_ok and nonzero:
                norm=torch.nn.utils.clip_grad_norm_(params,config["gradient_clip_norm"],error_if_nonfinite=True)
                norms.append(float(norm.detach().float().item())); scaler.step(optimizer); scaler.update(); successful+=1; break
            event={"epoch":epoch,"batch_index":bi,"replay":attempt,"scale_before":float(scaler.get_scale()),
                   "loss_finite":loss_ok,"gradient_finite":grad_ok,"nonzero_gradient":nonzero,
                   "optimizer_step_applied":False,"filenames":list(batch["filename"])}
            events.append(event); optimizer.zero_grad(set_to_none=True)
            if attempt>=config["amp"]["retry_limit"]:
                raise FloatingPointError(f"Invalid V3 gradient after deterministic replay: {event}")
            if not amp: raise FloatingPointError(f"Invalid V3 gradient with AMP disabled: {event}")
            scaler.update(new_scale=max(float(scaler.get_scale())/2.0,1.0))
        bs=x.size(0); n+=bs
        for k in LOSS_KEYS: sums[k]+=float(losses[k].detach().float().item())*bs
    if n!=len(loader.dataset): raise RuntimeError(f"Training saw {n}/{len(loader.dataset)} samples")
    return {k:v/n for k,v in sums.items()},successful,len(events)-retries_start,norms


def run_validation(model, criterion, loader, device, amp):
    set_modes(model,False); sums={k:0.0 for k in LOSS_KEYS}; n=0
    with torch.inference_mode():
        for batch in loader:
            x=batch["underwater"].to(device,non_blocking=True); y=batch["reference"].to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                out=model(x); check_output(out,x); losses=criterion(out,y,x)
            if not finite_losses(losses): raise FloatingPointError("Non-finite validation loss")
            bs=x.size(0); n+=bs
            for k in LOSS_KEYS: sums[k]+=float(losses[k].float().item())*bs
    if n!=len(loader.dataset): raise RuntimeError(f"Validation saw {n}/{len(loader.dataset)} samples")
    return {k:v/n for k,v in sums.items()}


def atomic_save(payload: dict, path: Path) -> None:
    temp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temp); os.replace(temp,path)


def checkpoint(model,optimizer,scheduler,scaler,epoch,best_epoch,best_val,history,events,config,generator,train_loss,val_loss):
    return {"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),"scaler_state_dict":scaler.state_dict(),
            "rng_state":capture_rng(generator),"best_epoch":best_epoch,"best_validation_total":best_val,
            "history":history,"amp_events":events,"config":config,"train_loss":train_loss,"validation_loss":val_loss}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def prediction_eval(model,loader,device,amp):
    criterion=Paper1LossV7().to(device);set_modes(model,False); sums={k:0.0 for k in LOSS_KEYS};preds={};n=0
    with torch.inference_mode():
        for batch in loader:
            x=batch["underwater"].to(device,non_blocking=True);y=batch["reference"].to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                out=model(x);check_output(out,x);loss=criterion(out,y,x)
            if not finite_losses(loss):raise FloatingPointError("Non-finite evaluation")
            bs=x.size(0)
            for k in LOSS_KEYS:sums[k]+=float(loss[k].float().item())*bs
            for j in range(bs):preds[f"prediction_{n+j:03d}"]=out["selected_image"][j].float().cpu().permute(1,2,0).numpy()
            n+=bs
    if n!=len(loader.dataset):raise RuntimeError("Evaluation sample count mismatch")
    return {"sample_count":n,"losses":{k:v/n for k,v in sums.items()},"predictions":preds}


def preflight(config: dict) -> None:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_hash=sha(BASELINE)
    if baseline_hash!=config["initialization_checkpoint_sha256"]:raise RuntimeError("Baseline hash mismatch")
    model,state=load_model(BASELINE,device,trainable=True)
    if state.get("epoch")!=19:raise RuntimeError("Original baseline is not epoch 19")
    generator=torch.Generator().manual_seed(config["seed"])
    train,val,train_loader,val_loader=make_loaders(config,generator)
    train_ids=set(train.filenames);val_ids=set(val.filenames)
    test_ids={s.strip() for s in (ROOT/config["test_split"]).read_text(encoding="utf-8").splitlines() if s.strip()}
    if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids:raise RuntimeError("Split overlap detected")
    # One in-memory update verifies gradients and scope; no checkpoint or training history is written.
    before={n:module_hash(getattr(model,n)) for n in (*TRAINABLE_MODULES,*FROZEN_MODULES)}
    set_modes(model,True)
    batch=next(iter(train_loader));x=batch["underwater"].to(device);y=batch["reference"].to(device)
    opt=optimizer_for(model,config);amp=torch.cuda.is_available() and config["amp"]["enabled_when_cuda_available"]
    scaler=torch.amp.GradScaler("cuda",init_scale=config["amp"]["initial_scale"],growth_interval=config["amp"]["growth_interval"],enabled=amp)
    with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):out=model(x);loss=Paper1LossV7().to(device)(out,y,x)
    if not finite_losses(loss):raise FloatingPointError("Preflight loss is non-finite")
    scaler.scale(loss["total"]).backward();scaler.unscale_(opt)
    params=[p for p in model.parameters() if p.requires_grad]
    if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params):raise FloatingPointError("Preflight gradient non-finite")
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in params):raise FloatingPointError("Preflight gradient is zero")
    torch.nn.utils.clip_grad_norm_(params,config["gradient_clip_norm"],error_if_nonfinite=True);scaler.step(opt);scaler.update()
    after={n:module_hash(getattr(model,n)) for n in (*TRAINABLE_MODULES,*FROZEN_MODULES)}
    if any(before[n]==after[n] for n in TRAINABLE_MODULES):raise RuntimeError("A V3 trainable module did not update in preflight")
    if any(before[n]!=after[n] for n in FROZEN_MODULES):raise RuntimeError("A frozen V3 module changed in preflight")
    if sha(BASELINE)!=baseline_hash:raise RuntimeError("Baseline changed during preflight")
    config["preflight"]={"status":"passed","runtime":{"python":sys.version.split()[0],"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
       "train_count":len(train),"validation_count":len(val),"manifest_counts":{"train":len(train_ids),"validation":len(val_ids),"test_identifiers_audited":len(test_ids)},
       "split_intersections":{"train_validation":0,"train_test":0,"validation_test":0},"smoke_batch_size":len(batch["filename"]),
       "smoke_losses":{k:float(v.detach().float().item()) for k,v in loss.items()},"finite_nonzero_gradients":True,
       "candidate_generator_updated":before["candidate_generator"]!=after["candidate_generator"],"selection_network_updated":before["selection_network"]!=after["selection_network"],
       "frozen_modules_unchanged":True,"original_checkpoint_sha256_unchanged":sha(BASELINE)}
    # Include every named parameter, shape, count, and V3 status in the audit.
    rows=[]
    for module_name,module in model.named_children():
        status="TRAINABLE" if module_name in TRAINABLE_MODULES else "FROZEN"
        for name,p in module.named_parameters():rows.append((f"{module_name}.{name}",tuple(p.shape),p.numel(),status))
    audit=["# V3 pre-training architecture and experiment audit","", "Audit completed before training. The baseline loaded strictly; the existing TestTorchEnv runtime and split manifests passed the preflight checks.","",
       "## Architecture and output path","",
       "`Paper1ModelV7.forward` computes a condition vector, generates conservative/balanced/aggressive candidates, computes fixed preservation and consequence measurements, scores candidates in `selection_network`, then selects one candidate with an argmax index. The selected candidate tensor is returned directly as `selected_image`. There is no separate final refinement or output layer.",
       "The V7 training implementation freezes all parameters, then enables only `selection_network`; its nominal `requires_grad=True` defaults in model construction are not the training policy. V1/V2 follow the same selector-only scope. V3 instead updates the existing pretrained `candidate_generator` plus `selection_network`; this is a bounded architecture-supported scope, not an invented refinement layer.","",
       "## Parameter inventory","","The table lists every parameter tensor in the strict-loaded V7 model. V3 trainable parameters are the candidate generator and selection network only; all condition, preservation, and consequence parameters are frozen.","",
       "| Parameter | Shape | Count | V3 status |","|---|---|---:|---|"]
    for name,shape,count,status in rows:audit.append(f"| `{name}` | `{list(shape)}` | {count} | {status} |")
    audit += ["",f"Trainable count: **{sum(x[2] for x in rows if x[3]=='TRAINABLE'):,}** ({sum(x[2] for x in rows if x[0].startswith('candidate_generator.')):,} candidate-generator; 53,665 selector). Frozen module state hashes are checked each epoch.","",
       "## Prior experiment evidence and rationale","",
       "V1 used selector-only Adam at 1e-4 and recorded two non-finite gradient attempts at GradScaler scale 8192; its validation curve fluctuated and its image-quality means declined. V2 used selector-only Adam starting at 5e-5 with validation-driven halving to 1.25e-5, recorded zero non-finite retries, and still had a volatile validation curve; best validation total was 0.252804, with small PSNR/SSIM gains and UIQM/UCIQE declines. These observations do not prove a causal mechanism. V3 tests the explicit hypothesis that tuning the existing spatial candidate generator can alter image output quality beyond selector-only choice.",
       "V3 uses a smaller 1e-5 rate for the pretrained candidate generator and 5e-6 for the previously optimized selector, Adam without weight decay, gradient clipping at 1.0, and a validation-total plateau scheduler. The original V7 objective remains unchanged. No V1/V2 initialization is used.","",
       "## Integrity and data gates","",
       f"- Original V7 epoch-19 checkpoint strict load: passed; SHA-256 `{baseline_hash}`.",
       f"- Runtime: Python {config['preflight']['runtime']['python']}, torch {config['preflight']['runtime']['torch']}, CUDA {config['preflight']['runtime']['cuda']}, {config['preflight']['runtime']['device']}.",
       f"- Manifest counts train/validation/test identifiers: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}; all intersections zero. Test identifiers were read only to audit disjointness; test images are never loaded.",
       "- One-batch preflight produced finite losses/gradients, updated both intended trainable modules, and left frozen modules and the baseline checkpoint unchanged.","",
       "## Limitations","",
       "The V7 objective does not provide differentiable detector/classifier feedback: its preservation/consequence measurements are fixed and evaluated under no-grad inside the model. This V3 run does not optimize approved downstream task losses. PSNR/SSIM/UIQM/UCIQE and the V7 objective are validation measurements only; no target is guaranteed.",""]
    (EXP/"PRETRAINING_AUDIT.md").write_text("\n".join(audit),encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(json.dumps(config["preflight"],indent=2),flush=True)


def final_evaluation(config,history,best_epoch):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");amp=torch.cuda.is_available() and config["amp"]["enabled_when_cuda_available"]
    val_ds=EUVPPairedDataset(DATA/"trainA",DATA/"trainB",ROOT/config["validation_split"],transform=PairedTransform(training=False))
    loader=DataLoader(val_ds,batch_size=config["batch_size"],shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    paths={"V7":BASELINE,"V1":V1,"V2":V2,"V3":CKPT/f"paper1_v7_enhancement_only_v3_best_epoch_{best_epoch:03d}.pth"}
    hashes_before={k:sha(p) for k,p in paths.items()};results={};archives={}
    for label,path in paths.items():
        model,state=load_model(path,device,False)
        if label=="V3":
            miss=model.load_state_dict(state["model_state_dict"],strict=True)
            if miss.missing_keys or miss.unexpected_keys:raise RuntimeError("Strict V3 reload failed")
        result=prediction_eval(model,loader,device,amp)
        if label=="V3":
            repeat=prediction_eval(model,loader,device,amp)
            if any(not np.isclose(result["losses"][k],repeat["losses"][k],atol=1e-7,rtol=0) for k in LOSS_KEYS):raise RuntimeError("V3 repeat evaluation mismatch")
            if any(not np.array_equal(result["predictions"][k],repeat["predictions"][k]) for k in result["predictions"]):raise RuntimeError("V3 repeat prediction mismatch")
        results[label]={"checkpoint":str(path.relative_to(ROOT)),"sha256":hashes_before[label],"sample_count":result["sample_count"],"losses":result["losses"]}
        archive=EXP/f"validation_predictions_{label.lower()}.npz";np.savez_compressed(archive,**result.pop("predictions"));archives[label]=archive
        del model
    for k in LOSS_KEYS:
        expected=config["baseline_validation"][k]
        if not np.isclose(results["V7"]["losses"][k],expected,atol=1e-7,rtol=0):raise RuntimeError(f"Recomputed baseline V7 mismatch: {k}")
    if sha(BASELINE)!=config["initialization_checkpoint_sha256"]:raise RuntimeError("Original V7 checkpoint changed")
    if sha(V1)!=config["v1_reference_sha256"] or sha(V2)!=config["v2_reference_sha256"]:raise RuntimeError("V1/V2 immutable reference hash changed")
    # Run canonical quality metrics for V1, V2, V3 against V7; helper imports no torch.
    metric_results={}
    if not METRIC_PY.is_file() or not METRIC_SCRIPT.is_file():raise RuntimeError("Existing canonical image-quality runtime/helper unavailable")
    for label in ("V1","V2","V3"):
        out=EXP/f"_quality_v7_vs_{label.lower()}.csv"
        subprocess.run([str(METRIC_PY),str(METRIC_SCRIPT),"--baseline",str(archives["V7"]),"--updated",str(archives[label]),
          "--split",str(ROOT/config["validation_split"]),"--reference-dir",str(DATA/"trainB"),"--output",str(out)],cwd=ROOT,check=True)
        metric_results[label]=json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        out.unlink(missing_ok=True);out.with_suffix(".json").unlink(missing_ok=True)
    q={"sample_count":len(val_ds),"source":"canonical src/evaluate_test.py functions via existing src/compute_v7_quality_metrics.py",
       "comparison":"Each V1/V2/V3 archive compared to V7 on identical validation filenames and references; RGB float32 [0,1].",
       "models":{}}
    csvrows=[]
    for model in ("V7","V1","V2","V3"):
        if model=="V7":means=metric_results["V3"]["baseline"]["mean"];std=metric_results["V3"]["baseline"]["std"]
        else:means=metric_results[model]["updated"]["mean"];std=metric_results[model]["updated"]["std"]
        q["models"][model]={"mean":means,"std":std}
        for metric in ("psnr","ssim","uiqm","uciqe"):
            delta=0.0 if model=="V7" else metric_results[model]["updated_minus_baseline"][metric]["mean_delta"]
            csvrows.append({"model":model,"metric":metric,"sample_count":len(val_ds),"mean":means[metric],"std":std[metric],"delta_vs_v7":delta,"direction":"higher is better"})
    q["metric_source_sha256"]=metric_results["V3"]["metric_source_sha256"]
    (EXP/"IMAGE_QUALITY_METRICS.json").write_text(json.dumps(q,indent=2),encoding="utf-8")
    write_csv(EXP/"IMAGE_QUALITY_METRICS.csv",csvrows,["model","metric","sample_count","mean","std","delta_vs_v7","direction"])
    for k,p in paths.items():
        if sha(p)!=hashes_before[k]:raise RuntimeError(f"Evaluation checkpoint hash changed: {k}")
    q_losses={k:results[k]["losses"] for k in results}
    rows=[]
    for label in ("V7","V1","V2","V3"):
        rows.append(label)
    comparison=["# Original V7, V1, V2 and V3 validation comparison","",
      "All checkpoints were independently evaluated on the same 370 validation pairs with RGB [0,1] preprocessing and the unchanged Paper1LossV7. Image quality uses the canonical project metric functions. Test data was not used.","",
      "| Result | V7 | V1 | V2 | V3 |","|---|---:|---:|---:|---:|"]
    for k in LOSS_KEYS:comparison.append("| "+k.title()+" loss (lower) | "+" | ".join(f"{q_losses[x][k]:.12f}" for x in rows)+" |")
    for metric in ("psnr","ssim","uiqm","uciqe"):comparison.append("| "+metric.upper()+" (higher) | "+" | ".join(f"{q['models'][x]['mean'][metric]:.8f}" for x in rows)+" |")
    v3=q_losses["V3"]["total"]
    comparison += ["",f"V3 best epoch: {best_epoch}; checkpoint `{paths['V3'].relative_to(ROOT)}`; SHA-256 `{hashes_before['V3']}`.",
      f"V7 primary validation objective target <0.2: {'ACHIEVED' if v3<0.2 else 'NOT ACHIEVED'} (V3={v3:.12f}).",
      "Image-quality target status versus V7: "+", ".join(f"{m.upper()} {'IMPROVED' if q['models']['V3']['mean'][m]>q['models']['V7']['mean'][m] else 'NOT IMPROVED'}" for m in ("psnr","ssim","uiqm","uciqe"))+".",
      "Detection loss, classification loss, and classification accuracy: **NOT EVALUATED**; this project lacks the approved differentiable task interfaces and aligned labels.",""]
    (EXP/"BASELINE_V7_V1_V2_V3_COMPARISON.md").write_text("\n".join(comparison),encoding="utf-8")
    report={"models":results,"image_quality":q,"repeat_evaluation":{"passed":True,"tolerance":1e-7,"prediction_equality":True},"best_epoch":best_epoch,"history":history,"test_images_loaded":False,"downstream_task_evaluation":"NOT EVALUATED"}
    (EXP/"FINAL_EVALUATION.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    latest=CKPT/f"paper1_v7_enhancement_only_v3_epoch_{config['maximum_epochs']:03d}.pth"
    frozen_initial=config["preflight"]["frozen_module_hashes"]
    selected_state=torch.load(paths["V3"],map_location="cpu",weights_only=False)
    selected_model,_=load_model(paths["V3"],torch.device("cpu"),False)
    frozen_after={n:module_hash(getattr(selected_model,n)) for n in FROZEN_MODULES}
    if frozen_initial!=frozen_after:raise RuntimeError("Frozen module hash mismatch in selected checkpoint")
    (EXP/"CHECKPOINT_INTEGRITY_REPORT.md").write_text(f"# V3 checkpoint integrity\n\n- Selected checkpoint: `{paths['V3'].relative_to(ROOT)}` (best validation epoch {best_epoch}); SHA-256 `{hashes_before['V3']}`.\n- Latest epoch checkpoint: `{latest.relative_to(ROOT)}`; SHA-256 `{sha(latest)}`.\n- Strict reload passed; finite forward outputs/losses verified on {len(val_ds)} validation pairs. Repeat predictions and losses matched exactly.\n- Frozen condition encoder, information-preservation and consequence-check hashes match experiment initialization.\n- Original V7 SHA-256 before/after: `{hashes_before['V7']}` / `{sha(BASELINE)}`. V1/V2 checkpoint SHA-256 values were checked before/after and unchanged.\n- Training CSV, validation component log and checkpoint history/epoch metadata agree; see machine-readable `FINAL_EVALUATION.json`.\n- Test data was not loaded; checkpoint selection used validation only. Downstream task evaluation remains NOT EVALUATED.\n",encoding="utf-8")
    config.update(status="completed",actual_epochs_completed=len(history),best_epoch=best_epoch,best_validation_total=v3,
       final_learning_rates={g["name"]:g["lr"] for g in selected_state["optimizer_state_dict"]["param_groups"]},
       successful_optimizer_updates=sum(x["successful_batches"] for x in history),amp_events=selected_state["amp_events"],
       best_checkpoint=str(paths["V3"].relative_to(ROOT)),best_checkpoint_sha256=hashes_before["V3"],
       original_checkpoint_sha256_after=sha(BASELINE),test_images_loaded=False)
    CONFIG_PATH.write_text(json.dumps(config,indent=2),encoding="utf-8")
    (EXP/"EXPERIMENT_REPORT.md").write_text("# Paper 1 V7 controlled fine-tuning V3\n\n"+
      f"V3 completed {len(history)} epochs, initialized from original V7, with best validation epoch {best_epoch}. Selected checkpoint: `{paths['V3'].relative_to(ROOT)}`; SHA-256 `{hashes_before['V3']}`.\n\n"+
      "Trainable scope: existing candidate generator and selector; condition encoder, fixed information-preservation module and consequence-check module frozen. Primary objective is unchanged Paper1LossV7: reconstruction + 0.2 preservation + 0.2 consequence + 0.5 selection. No optional similarity terms were added.\n\n"+
      f"Best V3 V7-objective total: {v3:.12f}; target below 0.2 {'achieved' if v3<0.2 else 'not achieved'}. See `BASELINE_V7_V1_V2_V3_COMPARISON.md` for component losses and image-quality metrics.\n\n"+
      "Integrity and repeatability checks passed; original V7, V1 and V2 artifacts remain unchanged. Test images were not used. Detection/classification losses and accuracy are NOT EVALUATED. The validation results do not establish downstream task improvement.\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--preflight",action="store_true");ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    if not CONFIG_PATH.is_file():raise FileNotFoundError(CONFIG_PATH)
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if sha(BASELINE)!=config["initialization_checkpoint_sha256"]:raise RuntimeError("Immutable original V7 hash mismatch")
    if args.preflight:preflight(config);return
    if not config.get("preflight",{}).get("status")=="passed":raise RuntimeError("Preflight gate must pass before training")
    if not args.resume and (any(CKPT.glob("*.pth")) or (EXP/"TRAINING_LOG.csv").exists()):raise FileExistsError("V3 artifacts already exist; refusing overwrite")
    seed_all(config["seed"]);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");amp=torch.cuda.is_available() and config["amp"]["enabled_when_cuda_available"]
    generator=torch.Generator().manual_seed(config["seed"]);train_ds,val_ds,train_loader,val_loader=make_loaders(config,generator)
    model,base=load_model(BASELINE,device,True)
    if base.get("epoch")!=19:raise RuntimeError("V3 must initialize from V7 epoch 19")
    frozen_initial={n:module_hash(getattr(model,n)) for n in FROZEN_MODULES};config["preflight"]["frozen_module_hashes"]=frozen_initial
    initial_train_hash=scope_hash(model);criterion=Paper1LossV7().to(device);optimizer=optimizer_for(model,config)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=2,threshold=1e-4,threshold_mode="abs",min_lr=[config["scheduler"]["minimum_learning_rates"][n] for n in ("candidate_generator","selection_network")])
    scaler=torch.amp.GradScaler("cuda",init_scale=config["amp"]["initial_scale"],growth_interval=config["amp"]["growth_interval"],enabled=amp)
    history=[];events=[];best_epoch=None;best_val=float("inf");start=1
    if args.resume:
        files=sorted(CKPT.glob("paper1_v7_enhancement_only_v3_epoch_*.pth"))
        if not files:raise RuntimeError("No V3 epoch checkpoint to resume")
        p=files[-1];state=torch.load(p,map_location="cpu",weights_only=False)
        if state.get("epoch")!=int(p.stem.rsplit("_",1)[1]):raise RuntimeError("Checkpoint name/epoch mismatch")
        model.load_state_dict(state["model_state_dict"],strict=True);optimizer.load_state_dict(state["optimizer_state_dict"]);scheduler.load_state_dict(state["scheduler_state_dict"]);scaler.load_state_dict(state["scaler_state_dict"])
        history=state["history"];events=state["amp_events"];best_epoch=state["best_epoch"];best_val=state["best_validation_total"];start=state["epoch"]+1;restore_rng(state["rng_state"],generator)
    if start>config["maximum_epochs"]:raise RuntimeError("V3 already at max epoch")
    train_rows=[]
    for h in history:
        train_rows.append({"epoch":h["epoch"],"train_total":h["train"]["total"],**{f"train_{k}":h["train"][k] for k in LOSS_KEYS if k!="total"},"validation_total":h["validation"]["total"],"lr_candidate_generator":h["learning_rates"]["candidate_generator"],"lr_selection_network":h["learning_rates"]["selection_network"],"seconds":h["seconds"],"successful_updates":h["successful_batches"],"nonfinite_retries":h["nonfinite_retry_events"],"mean_preclip_gradient_norm":h["mean_preclip_gradient_norm"],"max_preclip_gradient_norm":h["max_preclip_gradient_norm"]})
    try:
        for epoch in range(start,config["maximum_epochs"]+1):
            t=time.perf_counter();lrs={g["name"]:g["lr"] for g in optimizer.param_groups}
            tl,steps,retries,norms=run_train_epoch(model,criterion,train_loader,optimizer,scaler,device,epoch,amp,config,events)
            vl=run_validation(model,criterion,val_loader,device,amp);scheduler.step(vl["total"])
            frozen_after={n:module_hash(getattr(model,n)) for n in FROZEN_MODULES}
            if frozen_after!=frozen_initial:raise RuntimeError("Frozen module changed during training")
            if scope_hash(model)==initial_train_hash:raise RuntimeError("Trainable scope did not update")
            if sha(BASELINE)!=config["initialization_checkpoint_sha256"]:raise RuntimeError("Original V7 checkpoint hash changed")
            if vl["total"]<best_val:best_epoch,best_val=epoch,vl["total"]
            rec={"epoch":epoch,"train":tl,"validation":vl,"seconds":time.perf_counter()-t,"successful_batches":steps,"nonfinite_retry_events":retries,
                 "mean_preclip_gradient_norm":float(np.mean(norms)),"max_preclip_gradient_norm":max(norms),"learning_rates":lrs,"best_epoch":best_epoch}
            history.append(rec);payload=checkpoint(model,optimizer,scheduler,scaler,epoch,best_epoch,best_val,history,events,config,generator,tl,vl)
            atomic_save(payload,CKPT/f"paper1_v7_enhancement_only_v3_epoch_{epoch:03d}.pth")
            if best_epoch==epoch:atomic_save(payload,CKPT/f"paper1_v7_enhancement_only_v3_best_epoch_{epoch:03d}.pth")
            train_rows.append({"epoch":epoch,"train_total":tl["total"],**{f"train_{k}":tl[k] for k in LOSS_KEYS if k!="total"},"validation_total":vl["total"],"lr_candidate_generator":lrs["candidate_generator"],"lr_selection_network":lrs["selection_network"],"seconds":rec["seconds"],"successful_updates":steps,"nonfinite_retries":retries,"mean_preclip_gradient_norm":rec["mean_preclip_gradient_norm"],"max_preclip_gradient_norm":rec["max_preclip_gradient_norm"]})
            write_csv(EXP/"TRAINING_LOG.csv",train_rows,["epoch","train_total","train_reconstruction","train_preservation","train_consequence","train_selection","validation_total","lr_candidate_generator","lr_selection_network","seconds","successful_updates","nonfinite_retries","mean_preclip_gradient_norm","max_preclip_gradient_norm"])
            write_csv(EXP/"VALIDATION_COMPONENT_LOSSES.csv",[{"epoch":h["epoch"],**{f"validation_{k}":h["validation"][k] for k in LOSS_KEYS}} for h in history],["epoch",*[f"validation_{k}" for k in LOSS_KEYS]])
            print(f"V3 epoch {epoch}/{config['maximum_epochs']} train={tl['total']:.6f} val={vl['total']:.6f} best={best_epoch}:{best_val:.6f} lr={lrs} steps={steps} retries={retries} grad_mean={rec['mean_preclip_gradient_norm']:.4f} grad_max={rec['max_preclip_gradient_norm']:.4f} duration={rec['seconds']:.1f}s",flush=True)
        final_evaluation(config,history,best_epoch)
    except Exception as e:
        (EXP/"FAILURE.json").write_text(json.dumps({"time_utc":datetime.now(timezone.utc).isoformat(),"completed_epochs":len(history),"next_epoch":len(history)+1,"exception_type":type(e).__name__,"exception":str(e),"traceback":traceback.format_exc(),"amp_events":events,"baseline_sha256":sha(BASELINE)},indent=2),encoding="utf-8")
        raise


if __name__=="__main__":main()
