# V5 Training Report

Decision: **V5 NEEDS REFINEMENT**

## Objective

argmin_c mean_k((L_k(c)-median_train_k)/IQR_train_k), k in {reference L1, 1-preservation score, consequence reconstruction L1}; equal 1/3 weights.

Normalization uses component medians and IQRs fitted once on train.txt. Validation and test data do not fit these values.

## Training and validation

Best epoch: 12.

## Train-only target distribution

Calibration used 2,960 training images before V5 optimization.

| Candidate | V4 reference-L1 target | V5 multi-objective target |
|---|---:|---:|
| Conservative | 9 (0.30%) | 28 (0.95%) |
| Balanced | 20 (0.68%) | 228 (7.70%) |
| Aggressive | 2931 (99.02%) | 2704 (91.35%) |

| Epoch | Train total | Val total | Train selection | Val selection |
|---:|---:|---:|---:|---:|
| 1 | 0.272711 | 0.158981 | 0.364476 | 0.155782 |
| 2 | 0.131955 | 0.102598 | 0.095492 | 0.057707 |
| 3 | 0.123820 | 0.159740 | 0.092351 | 0.141424 |
| 4 | 0.373181 | 0.319636 | 0.571227 | 0.464289 |
| 5 | 0.230217 | 0.179518 | 0.282204 | 0.182811 |
| 6 | 0.164611 | 0.098296 | 0.152925 | 0.022623 |
| 7 | 0.101902 | 0.093928 | 0.032310 | 0.021023 |
| 8 | 0.141978 | 0.359664 | 0.114911 | 0.536907 |
| 9 | 0.240460 | 0.165056 | 0.304870 | 0.159361 |
| 10 | 0.169673 | 0.122435 | 0.170838 | 0.081391 |
| 11 | 0.122214 | 0.113607 | 0.081175 | 0.065777 |
| 12 | 0.146373 | 0.082382 | 0.122569 | 0.002004 |
| 13 | 0.138179 | 0.201269 | 0.112772 | 0.231065 |
| 14 | 0.113046 | 0.179101 | 0.061380 | 0.188200 |
| 15 | 0.100807 | 0.107312 | 0.040045 | 0.056162 |
| 16 | 0.150631 | 0.168907 | 0.137840 | 0.187380 |
| 17 | 0.201097 | 0.366696 | 0.236515 | 0.536374 |
| 18 | 0.116012 | 0.102399 | 0.040527 | 0.013170 |
| 19 | 0.123448 | 0.127509 | 0.056995 | 0.067203 |
| 20 | 0.125804 | 0.144836 | 0.062346 | 0.099603 |

## Validation comparison

```json
{
  "sample_count": 370,
  "v4_selection_distribution": {
    "Conservative": 0,
    "Balanced": 0,
    "Aggressive": 370
  },
  "v5_target_distribution": {
    "Conservative": 0,
    "Balanced": 370,
    "Aggressive": 0
  },
  "v5_selection_distribution": {
    "Conservative": 0,
    "Balanced": 370,
    "Aggressive": 0
  },
  "v4_selector_to_v5_target_confusion": {
    "Conservative": {
      "Conservative": 0,
      "Balanced": 1,
      "Aggressive": 0
    },
    "Balanced": {
      "Conservative": 0,
      "Balanced": 3,
      "Aggressive": 0
    },
    "Aggressive": {
      "Conservative": 0,
      "Balanced": 366,
      "Aggressive": 0
    }
  },
  "v5_target_to_v5_selector_confusion": {
    "Conservative": {
      "Conservative": 0,
      "Balanced": 0,
      "Aggressive": 0
    },
    "Balanced": {
      "Conservative": 0,
      "Balanced": 370,
      "Aggressive": 0
    },
    "Aggressive": {
      "Conservative": 0,
      "Balanced": 0,
      "Aggressive": 0
    }
  },
  "v4_selector_to_v5_selector_confusion": {
    "Conservative": {
      "Conservative": 0,
      "Balanced": 0,
      "Aggressive": 0
    },
    "Balanced": {
      "Conservative": 0,
      "Balanced": 0,
      "Aggressive": 0
    },
    "Aggressive": {
      "Conservative": 0,
      "Balanced": 370,
      "Aggressive": 0
    }
  },
  "v5_target_selector_agreement": 1.0,
  "v4_target_to_v5_target_confusion": {
    "Conservative": {
      "Conservative": 0,
      "Balanced": 1,
      "Aggressive": 0
    },
    "Balanced": {
      "Conservative": 0,
      "Balanced": 3,
      "Aggressive": 0
    },
    "Aggressive": {
      "Conservative": 0,
      "Balanced": 366,
      "Aggressive": 0
    }
  },
  "mean_selection_entropy": 0.007931932397273281,
  "mean_utility_margin": 9.370975940291945,
  "mean_v5_selection_probabilities": {
    "Conservative": 9.150318620856057e-05,
    "Balanced": 0.9980996077125137,
    "Aggressive": 0.001808885519983899
  },
  "selection_distribution_percentages": {
    "v4": {
      "Conservative": 0.0,
      "Balanced": 0.0,
      "Aggressive": 100.0
    },
    "v5_target": {
      "Conservative": 0.0,
      "Balanced": 100.0,
      "Aggressive": 0.0
    },
    "v5_selector": {
      "Conservative": 0.0,
      "Balanced": 100.0,
      "Aggressive": 0.0
    }
  },
  "mean_v4_metrics": {
    "psnr": 23.372510954363555,
    "ssim": 0.8132755376197196,
    "uiqm": 1.7115460476864606,
    "uciqe": 5.873283769311132
  },
  "mean_v5_metrics": {
    "psnr": 21.65962939624159,
    "ssim": 0.7963453898558746,
    "uiqm": 1.668473452606055,
    "uciqe": 5.825829986623815
  },
  "adaptivity_by_input_reference_l1_quartile": {
    "input_reference_l1_q1": {
      "Conservative": 0,
      "Balanced": 93,
      "Aggressive": 0
    },
    "input_reference_l1_q2": {
      "Conservative": 0,
      "Balanced": 92,
      "Aggressive": 0
    },
    "input_reference_l1_q3": {
      "Conservative": 0,
      "Balanced": 92,
      "Aggressive": 0
    },
    "input_reference_l1_q4": {
      "Conservative": 0,
      "Balanced": 93,
      "Aggressive": 0
    }
  },
  "median_input_reference_l1_quartile_edges": [
    0.047843124717473984,
    0.09782713651657104,
    0.11644341424107552,
    0.13583612069487572,
    0.302097350358963
  ],
  "mean_v4_selection_objectives": {
    "reference_l1": 0.05492515142604306,
    "preservation_distortion": 0.04501686805003398,
    "consequence": 0.05231502394418459,
    "residual_l2": 53.85591207968222
  },
  "mean_v5_selection_objectives": {
    "reference_l1": 0.06659764168733681,
    "preservation_distortion": 0.033660485776695045,
    "consequence": 0.04110551217973635,
    "residual_l2": 43.04733972807188
  },
  "v5_candidate_diversity": {
    "pairwise_mean_absolute_candidate_distance": {
      "conservative_vs_balanced": 0.15696812783544128,
      "balanced_vs_aggressive": 0.1605483908508275,
      "conservative_vs_aggressive": 0.07559696871887993
    },
    "pairwise_mean_residual_cosine_similarity": {
      "conservative_vs_balanced": -0.926165801447791,
      "balanced_vs_aggressive": -0.5548282229184248,
      "conservative_vs_aggressive": 0.6150608780592478
    },
    "mean_residual_l2_norm": {
      "conservative": 29.439115436657055,
      "balanced": 43.04733966105693,
      "aggressive": 49.7237963135178
    },
    "mean_preservation_score": {
      "conservative": 0.9681426697486156,
      "balanced": 0.966339514223305,
      "aggressive": 0.9692555769069775
    },
    "validation_candidate_quality": {
      "conservative": {
        "psnr": 14.052992407310706,
        "ssim": 0.6408094082329724
      },
      "balanced": {
        "psnr": 21.65962939624159,
        "ssim": 0.7963453898558746
      },
      "aggressive": {
        "psnr": 13.32434983509209,
        "ssim": 0.6398835617142755
      }
    }
  },
  "v4_candidate_diversity": {
    "pairwise_mean_absolute_candidate_distance": {
      "conservative_vs_balanced": 0.011103761389322982,
      "balanced_vs_aggressive": 0.10783049738084947,
      "conservative_vs_aggressive": 0.1136238351564955
    },
    "pairwise_mean_residual_cosine_similarity": {
      "conservative_vs_balanced": 0.7598270161329089,
      "balanced_vs_aggressive": 0.3440523687146,
      "conservative_vs_aggressive": -0.04631934551887114
    },
    "mean_residual_l2_norm": {
      "conservative": 5.525165682547801,
      "balanced": 9.352000686284658,
      "aggressive": 53.85591213123219
    },
    "mean_preservation_score": {
      "conservative": 0.9967579421159384,
      "balanced": 0.9958794369890883,
      "aggressive": 0.9549831335609024
    },
    "validation_candidate_quality": {
      "conservative": {
        "psnr": 16.45674641841389,
        "ssim": 0.7191071776119439
      },
      "balanced": {
        "psnr": 16.781357010623328,
        "ssim": 0.7311450560350676
      },
      "aggressive": {
        "psnr": 23.372510954363555,
        "ssim": 0.8132755376197196
      }
    }
  }
}
```

The selection groups use validation input-to-reference L1 quartiles as a measurable degradation proxy; this is association, not causal evidence.

Per-image diagnostics: `diagnostics\v5_selection\validation_per_image.json`.

Two selection-gradient overflows were recorded (epoch 1 batch 182 and epoch 13 batch 321). On both batches, selection-component gradients were non-finite at AMP 8,192 and 4,096, and finite at 2,048, 1,024, 512, and FP32; neither failed optimizer step was applied. The JSON report includes filenames and per-scale diagnostics.

No held-out test images were read or evaluated. Final held-out testing is not authorized by this report.