"""V6 loss: fixed fidelity eligibility, then preservation/consequence preference."""
import torch
import torch.nn.functional as F
from paper1_loss import Paper1Loss

class Paper1LossV6(Paper1Loss):
    names=("conservative","balanced","aggressive")
    def __init__(self, calibration):
        super().__init__()
        self.register_buffer("median",torch.as_tensor(calibration["median"],dtype=torch.float32))
        self.register_buffer("iqr",torch.as_tensor(calibration["iqr"],dtype=torch.float32))
        self.fidelity_tolerance=float(calibration["relative_fidelity_tolerance"])
        if self.fidelity_tolerance < 0 or self.iqr.numel()!=2 or torch.any(self.iqr<=0):
            raise ValueError("Invalid frozen V6 threshold/ranking calibration")
    def candidate_targets(self,outputs,reference):
        cand=outputs["candidates"]
        ref=torch.stack([F.l1_loss(cand[n],reference,reduction="none").mean((1,2,3)) for n in self.names],1)
        pres=torch.stack([1.0-outputs["preservation_scores"][n] for n in self.names],1)
        cons=torch.stack([outputs["candidate_consequence_errors"][n] for n in self.names],1)
        best=ref.min(1,keepdim=True).values
        gaps=(ref-best)/(best+1e-8)
        eligible=gaps<=self.fidelity_tolerance
        raw=torch.stack((pres,cons),2)
        z=(raw-self.median.view(1,1,2))/self.iqr.view(1,1,2)
        preference=z.mean(2)
        masked=preference.masked_fill(~eligible,float("inf"))
        targets=masked.argmin(1)
        return targets,{"reference":ref,"preservation":pres,"consequence":cons,"relative_fidelity_gap":gaps,"eligible":eligible,"preference":preference,"score":masked}
    def forward(self,outputs,reference,original):
        cand=outputs["candidates"];weights=outputs["selection_weights"]
        lref=torch.stack([F.l1_loss(cand[n],reference,reduction="none").mean((1,2,3)) for n in self.names],1)
        reconstruction=(weights*lref).sum(1).mean()
        preservation=torch.stack([1.0-outputs["preservation_scores"][n] for n in self.names],1).mean()
        consequence=F.l1_loss(outputs["reconstructed"],original)
        targets,_=self.candidate_targets(outputs,reference)
        selection=F.cross_entropy(outputs["utilities"],targets)
        total=self.lambda_recon*reconstruction+self.lambda_preserve*preservation+self.lambda_consequence*consequence+self.lambda_selection*selection
        return {"total":total,"reconstruction":reconstruction,"preservation":preservation,"consequence":consequence,"selection":selection}
