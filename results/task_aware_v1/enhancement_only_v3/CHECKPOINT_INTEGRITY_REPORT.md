# V3 checkpoint integrity

- Selected checkpoint: `checkpoints\v7_enhancement_only_v3\paper1_v7_enhancement_only_v3_best_epoch_009.pth` (best validation epoch 9); SHA-256 `29F99549FBE2ED56ED9B56B4F3F4A34A82A6839AD0678D2F2A4DD91BD7A51EFA`.
- Latest epoch checkpoint: `checkpoints\v7_enhancement_only_v3\paper1_v7_enhancement_only_v3_epoch_020.pth`; SHA-256 `33E5B0EF7172B6E6DE1229C7C6A2E966E18141A852A429C1D9509F22CDB55CAF`.
- Strict reload passed; finite forward outputs/losses verified on 370 validation pairs. Repeat predictions and losses matched exactly.
- Frozen condition encoder, information-preservation and consequence-check hashes match experiment initialization.
- Original V7 SHA-256 before/after: `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E` / `A0E04FEEC9A6C69CCEE0DD8A6E23E9C35227CDAA10C270DD82F853536920B49E`. V1/V2 checkpoint SHA-256 values were checked before/after and unchanged.
- Training CSV, validation component log and checkpoint history/epoch metadata agree; see machine-readable `FINAL_EVALUATION.json`.
- Test data was not loaded; checkpoint selection used validation only. Downstream task evaluation remains NOT EVALUATED.
