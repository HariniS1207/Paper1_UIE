"""Validation-only complete diagnosis and representative visual panels."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "src/data", ROOT / "src/models", ROOT / "src/losses"):
    sys.path.insert(0, str(p))
from euvp_dataset import EUVPPairedDataset
from transforms import PairedTransform
from models.paper1_model_final_selector import Paper1ModelFinalSelector
from losses.paper1_loss_final_selector import Paper1LossFinalSelector
from evaluate_test import calculate_metrics

DATA = ROOT / "data/EUVP/EUVP-Dataset/EUVP/Paired/underwater_imagenet"
NAMES = ("Conservative", "Balanced", "Aggressive")
LOWER = tuple(n.lower() for n in NAMES)
EDGES = [0.047843124717473984, 0.09782713651657104, 0.11644341424107552,
         0.13583612069487572, 0.302097350358963]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest().upper()

def metrics(image, ref):
    a=image.detach().float().cpu().permute(1,2,0).numpy()
    b=ref.detach().float().cpu().permute(1,2,0).numpy()
    vals=calculate_metrics(a,b)
    return dict(zip(("psnr","ssim","uiqm","uciqe"),map(float,vals)))

def main():
    ckptdir=ROOT/"checkpoints/final_selector"
    history=json.loads((ckptdir/"loss_history.json").read_text(encoding="utf-8"))
    best=min(history,key=lambda r:r["validation"]["total"]); epoch=best["epoch"]
    ckpt=ckptdir/f"final_selector_best_epoch_{epoch:03d}.pth"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=EUVPPairedDataset(DATA/"trainA",DATA/"trainB",ROOT/"data/splits/val.txt",transform=PairedTransform(training=False))
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    model=Paper1ModelFinalSelector().to(device)
    state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); model.eval()
    criterion=Paper1LossFinalSelector(); records=[]; representative_tensors={}
    for batch in tqdm(loader,desc="Final selector validation only"):
        x,ref=batch["underwater"].to(device),batch["reference"].to(device)
        with torch.inference_mode():
            out=model(x); objectives=criterion.candidate_objectives(out,ref); targets,rankinfo=criterion.soft_targets(objectives)
        for i,filename in enumerate(batch["filename"]):
            sev=float((x[i]-ref[i]).abs().mean())
            probs=out["selection_weights"][i].float().cpu().tolist(); ix=int(out["selected_index"][i])
            cand={}
            for k,name in enumerate(NAMES):
                low=name.lower(); cm=metrics(out["candidates"][low][i],ref[i])
                cand[name]={"fidelity_l1":float(objectives[i,k,0]),"preservation_score":float(out["preservation_scores"][low][i]),
                    "preservation_distortion":float(objectives[i,k,1]),"consequence_error":float(objectives[i,k,2]),**cm}
            ent=float(-(torch.tensor(probs).clamp_min(1e-12)*torch.tensor(probs).clamp_min(1e-12).log()).sum())
            util=out["utilities"][i].float().detach().cpu().sort(descending=True).values
            record={"filename":filename,"severity_input_reference_l1":sev,
                "condition_vector":out["condition_vector"][i].float().cpu().tolist(),
                "probabilities":dict(zip(NAMES,probs)),"selected":NAMES[ix],"selected_index":ix,
                "target_soft":dict(zip(NAMES,targets[i].float().cpu().tolist())),"target_hard":NAMES[int(targets[i].argmax())],
                "mean_rank":dict(zip(NAMES,rankinfo["mean_rank"][i].float().cpu().tolist())),
                "selection_entropy_nats":ent,"utility_margin":float(util[0]-util[1]),
                "candidate_metrics":cand,"selected_metrics":metrics(out["selected_image"][i],ref[i]),
                "original_metrics":metrics(x[i],ref[i]),
                "selected_preservation_score":cand[NAMES[ix]]["preservation_score"],
                "selected_consequence_error":cand[NAMES[ix]]["consequence_error"]}
            records.append(record)
            score=max(probs)
            prev=representative_tensors.get(NAMES[ix])
            if prev is None or score>prev["score"]:
                representative_tensors[NAMES[ix]]={"score":score,"filename":filename,
                    "images":[x[i].detach().cpu(),*[out["candidates"][n.lower()][i].detach().cpu() for n in NAMES],
                              out["selected_image"][i].detach().cpu()],"record":record}
    counts={n:sum(r["selected"]==n for r in records) for n in NAMES}
    target_counts={n:sum(r["target_hard"]==n for r in records) for n in NAMES}
    target_mass={n:float(np.mean([r["target_soft"][n] for r in records])) for n in NAMES}
    confusion={t:{s:0 for s in NAMES} for t in NAMES}
    for r in records: confusion[r["target_hard"]][r["selected"]]+=1
    groups=[]
    for q in range(4):
        subset=[r for r in records if EDGES[q] <= r["severity_input_reference_l1"] and
                (r["severity_input_reference_l1"] < EDGES[q+1] or q==3)]
        groups.append({"quartile":f"Q{q+1}","n":len(subset),"mean_severity":float(np.mean([r["severity_input_reference_l1"] for r in subset])),
            "selected_counts":{n:sum(r["selected"]==n for r in subset) for n in NAMES},
            "selected_percentages":{n:100*sum(r["selected"]==n for r in subset)/len(subset) for n in NAMES},
            "mean_entropy_nats":float(np.mean([r["selection_entropy_nats"] for r in subset])),
            "mean_confidence":float(np.mean([max(r["probabilities"].values()) for r in subset])),
            "mean_utility_margin":float(np.mean([r["utility_margin"] for r in subset]))})
    def mean_metric(rs,key): return {m:float(np.mean([r[key][m] for r in rs])) for m in ("psnr","ssim","uiqm","uciqe")}
    selected_metrics=mean_metric(records,"selected_metrics"); original_metrics=mean_metric(records,"original_metrics")
    selected_pres=float(np.mean([r["selected_preservation_score"] for r in records]))
    selected_cons=float(np.mean([r["selected_consequence_error"] for r in records]))
    candidate_summary={n:{k:float(np.mean([r["candidate_metrics"][n][k] for r in records])) for k in
        ("fidelity_l1","preservation_score","consequence_error","psnr","ssim","uiqm","uciqe")} for n in NAMES}
    v7=json.loads((ROOT/"diagnostics/v7_selection/v7_selector_diagnosis.json").read_text(encoding="utf-8"))
    comparisons={"FinalSelector":{"selected_metrics":selected_metrics,"preservation_score":selected_pres,"consequence_error":selected_cons},
                "V7":v7["comparison_v3_v4_v5_v6_v7_validation_only"]["V7"]}
    for v in ("V3","V4","V5","V6"):
        comparisons[v]=v7["comparison_v3_v4_v5_v6_v7_validation_only"][v]
    reps={}
    for name in NAMES:
        eligible=[r for r in records if r["selected"]==name]
        reps[name]=sorted(eligible,key=lambda r:(max(r["probabilities"].values()),r["utility_margin"]),reverse=True)[:5]
    result={"split":"validation only","sample_count":len(records),"best_epoch":epoch,"best_validation_loss":best["validation"]["total"],
        "checkpoint":str(ckpt.relative_to(ROOT)),"checkpoint_sha256":sha(ckpt),"target_hard_counts":target_counts,
        "target_mean_soft_distribution":target_mass,"selector_counts":counts,"target_selector_agreement":float(np.mean([r["target_hard"]==r["selected"] for r in records])),
        "confusion_matrix":confusion,"mean_selection_probabilities":{n:float(np.mean([r["probabilities"][n] for r in records])) for n in NAMES},
        "mean_selection_entropy_nats":float(np.mean([r["selection_entropy_nats"] for r in records])),
        "mean_utility_margin":float(np.mean([r["utility_margin"] for r in records])),"severity_quartile_edges":EDGES,
        "severity_groups":groups,"candidate_diversity_from_frozen_v7_candidate_space":v7["candidate_diversity"],
        "candidate_quality":candidate_summary,"selected_metrics":selected_metrics,"original_metrics":original_metrics,
        "selected_metric_improvements_over_original":{k:selected_metrics[k]-original_metrics[k] for k in selected_metrics},
        "selected_preservation_score":selected_pres,"selected_consequence_error":selected_cons,
        "comparisons_v3_v4_v5_v6_v7_validation_only":comparisons,"representative_examples_by_selected_candidate":reps,
        "amp_events":state.get("amp_events",[]),"test_split_loaded":False}
    output_paths=[ROOT/"diagnostics/final_selector/selector_diagnosis.json",ROOT/"diagnostics/final_selector/selector_diagnosis.md",
        ROOT/"diagnostics/final_selector/validation_per_image.json",ROOT/"results/final_selector/training_report.json",ROOT/"results/final_selector/training_report.md"]
    if any(p.exists() for p in output_paths): raise FileExistsError("Refusing to overwrite existing final-selector reports")
    (ROOT/"diagnostics/final_selector/validation_per_image.json").write_text(json.dumps({"split":"validation","records":records},indent=2),encoding="utf-8")
    (ROOT/"diagnostics/final_selector/selector_diagnosis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    config=json.loads((ckptdir/"training_config.json").read_text(encoding="utf-8"))
    (ROOT/"results/final_selector/training_report.json").write_text(json.dumps({"training_config":config,"history":history,"diagnosis":result,"test_set_accessed":False},indent=2),encoding="utf-8")
    # Markdown report is generated from the same saved validation measurements.
    lines=["# Final condition-aware selector report","","## Validation decision", "", "Pending full metric/selection review; see conclusion after validation analysis.","",
        f"- Best epoch: {epoch}; validation total loss: {best['validation']['total']:.6f}.",f"- Checkpoint SHA-256: `{result['checkpoint_sha256']}`.","- Test set accessed: **NO**.","",
        "## Validation metrics","","| Model | PSNR | SSIM | UIQM | UCIQE | Preservation | Consequence |","|---|---:|---:|---:|---:|---:|---:|"]
    for v in ("FinalSelector","V3","V4","V5","V6","V7"):
        x=comparisons[v]; m=x.get("selected_metrics",{}); pr=x.get("preservation_score",""); co=x.get("consequence_error","")
        if v in ("V5","V6"):
            q=x.get("selected_preservation_distortion_and_consequence",{}); pr=1-float(q.get("preservation_distortion",0)); co=q.get("consequence_error","")
        if v=="V3": pr=v7["selected_preservation_consequence"]["V3"]["preservation_score"]; co=v7["selected_preservation_consequence"]["V3"]["consequence_error"]
        lines.append(f"| {v} | {m.get('psnr',float('nan')):.4f} | {m.get('ssim',float('nan')):.4f} | {m.get('uiqm',float('nan')):.4f} | {m.get('uciqe',float('nan')):.4f} | {pr} | {co} |")
    lines += ["","## Selector behavior","",f"- Hard target counts: {target_counts}; mean soft target mass: {target_mass}.",f"- Selected counts: {counts}; target agreement: {result['target_selector_agreement']:.3%}.",f"- Mean probabilities: {result['mean_selection_probabilities']}.",f"- Mean entropy: {result['mean_selection_entropy_nats']:.4f} nats; mean utility margin: {result['mean_utility_margin']:.4f}.","","| Quartile | Mean severity | C | B | A | Mean entropy | Mean confidence | Mean margin |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        p=g["selected_percentages"]; lines.append(f"| {g['quartile']} | {g['mean_severity']:.4f} | {p[NAMES[0]]:.1f}% | {p[NAMES[1]]:.1f}% | {p[NAMES[2]]:.1f}% | {g['mean_entropy_nats']:.4f} | {g['mean_confidence']:.4f} | {g['mean_utility_margin']:.4f} |")
    lines += ["","### Representative selected examples","","| Selected class | Image | Probabilities (C/B/A) |","|---|---|---|"]
    for n, rs in reps.items():
        if rs:
            r=rs[0]; p=r["probabilities"]; lines.append(f"| {n} | `{r['filename']}` | {p[NAMES[0]]:.3f}/{p[NAMES[1]]:.3f}/{p[NAMES[2]]:.3f} |")
        else: lines.append(f"| {n} | No validation image selected this class | - |")
    if all(n in representative_tensors for n in NAMES):
        panel_dir=ROOT/"results/final_selector/representative_examples"
        panel_dir.mkdir(exist_ok=False)
        titles=("Original","Conservative","Balanced","Aggressive","Selected")
        for name, item in representative_tensors.items():
            images=[]
            for tensor,title in zip(item["images"],titles):
                arr=(tensor.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                images.append(Image.fromarray(arr))
            w,h=images[0].size; panel=Image.new("RGB",(w*5,h+72),"white"); draw=ImageDraw.Draw(panel)
            rec=item["record"]; probs=rec["probabilities"]
            header=f"{rec['filename']} | selected {name} | C/B/A={probs[NAMES[0]]:.3f}/{probs[NAMES[1]]:.3f}/{probs[NAMES[2]]:.3f} | preservation={rec['selected_preservation_score']:.4f}"
            draw.text((8,6),header,fill="black")
            for j,(im,title) in enumerate(zip(images,titles)):
                panel.paste(im,(j*w,32)); draw.text((j*w+6,32+h+5),title,fill="black")
            panel.save(panel_dir/f"{name.lower()}_selected_{Path(rec['filename']).stem}.png")
    (ROOT/"results/final_selector/training_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Diagnosis markdown summarizes independent adaptivity evidence.
    diag=["# Final selector validation diagnosis","","- Model: explicit per-candidate bilinear condition-candidate compatibility.",
        f"- Selected counts: {counts}; target counts: {target_counts}.",f"- Selected probabilities mean: {result['mean_selection_probabilities']}.",
        f"- Target-selector agreement: {result['target_selector_agreement']:.3%}.",f"- Mean entropy: {result['mean_selection_entropy_nats']:.4f}; margin: {result['mean_utility_margin']:.4f}.","",
        "| Quartile | Severity | C | B | A | Entropy | Confidence | Margin |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        p=g["selected_percentages"]; diag.append(f"| {g['quartile']} | {g['mean_severity']:.4f} | {p[NAMES[0]]:.1f}% | {p[NAMES[1]]:.1f}% | {p[NAMES[2]]:.1f}% | {g['mean_entropy_nats']:.4f} | {g['mean_confidence']:.4f} | {g['mean_utility_margin']:.4f} |")
    (ROOT/"diagnostics/final_selector/selector_diagnosis.md").write_text("\n".join(diag)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
