# Public brain MRI dataset survey for groupwise rigid/affine/SVF registration (MIDL 2027)

Survey date: 2026-10-12. Research was done only with web search and fetch. Nothing was downloaded. Each claim has a source link. **UNVERIFIED** means I could not confirm it from a primary source during this survey. Source text was rephrased for licensing compliance.

## 1. Executive summary

You can build the whole MIDL plan from open or quickly approved data. Four access requests are worth sending today:

1. **OASIS (all sets).** One form covers OASIS-1, OASIS-2 and OASIS-3. You need the raw OASIS-1/2 data for its same-session repeat scans and the OASIS-3 T2/FLAIR for MultiMorph parity.
2. **ADNI.** Review is usually within about 2 weeks.
3. **MIRIAD.** Online DUA.
4. **Cam-CAN.** Online DUA.

Skip UK Biobank, ABCD, HCP-Aging/Development (NDA) and SRPBS. Each needs a fee, an institutional signing official, or slow review, and none adds anything a faster dataset can't provide.

**IXI is the shared test set** for every learned baseline: MultiMorph, BrainMorph, KeyMorph and SynthMorph all report on it. It is open (CC BY-SA 3.0) and has T1/T2/PD. It must be **fully excluded from training**. That also rules out OpenBHB/LUMIR as training data, because OpenBHB contains IXI and MPI-Leipzig.

Proposed split:
- **Training:** Neurite-OASIS (minus the subjects whose IDs collide with the manual-label test sets) plus OASIS-3 and synthetic images.
- **Longitudinal rigid test:** ADNI + MIRIAD, the exact protocol of arXiv 2501.13010, plus OASIS-2 for BrainMorph's protocol.
- **Exact-ground-truth multi-contrast rigid test:** intrinsically co-registered contrasts, i.e. IXI PD/T2 dual-echo and MP2RAGE INV1/INV2/UNI, with synthetic rigid transforms applied on top.
- **Manual-label deformable test:** Mindboggle-101 and LPBA40.
- **Atlas / groupwise test:** IXI-Guy's, the MultiMorph protocol.

On your Neurite-OASIS question: you are **partly right**. The `norm` images come from FreeSurfer's skull-strip and bias-correction pipeline. FreeSurfer's intensity normalization puts white matter at about 110. The release then scales everything to [0, 1]. The `aligned_*` volumes sit in **FreeSurfer Talairach (MNI305) space via `talairach.xfm`, not MNI152**.

## 2. Answers to A and B

### A. Neurite-OASIS (Hoopes & Dalca, used in HyperMorph and Learn2Reg 2021)

| Item | Finding | Source |
|---|---|---|
| Subjects | 414 OASIS-1 subjects (the full OASIS-1 release has 416) | [neurite-oasis.md](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) |
| Processing | "Processed with FreeSurfer and SAMSEG." `norm` = FreeSurfer skull-stripped + bias-corrected. All images conformed to one shape and scaled to [0,1] | same |
| Intensity | FreeSurfer normalization puts WM values around 110 ([mri_normalize docs](https://surfer.nmr.mgh.harvard.edu/pub/docs/wiki/mri_normalize.help.xml.html)). Neurite-OASIS then rescales to [0,1]. So `norm` is WM-peak normalized and then rescaled, which is not a plain min-max of raw data. Whether `norm.nii.gz` is the FreeSurfer `norm.mgz` or `brain.mgz`, and the exact scaling divisor (max or 255), are **UNVERIFIED**. Check on disk: the WM mode should be roughly constant across subjects | same |
| Files per subject | `orig.nii.gz` and `norm.nii.gz` (256³, scanner space); `aligned_orig` and `aligned_norm` (160×192×224, template space); `slice_orig` and `slice_norm` (2D coronal slice 109, 160×192); segmentations `seg35`, `seg4` and 2D `seg24`, with FreeSurfer LUT files `seg35_labels.txt` etc. | same |
| Template space | Resampled into **FreeSurfer Talairach space with the recon-all `talairach.xfm`** (MNI305 affine), cropped by [(48,48),(31,33),(3,29)] | same |
| Segmentations | Automated: a 35-label major-structure map, a 4-label tissue map derived from it, and a 24-label 2D map. Whether seg35 comes from SAMSEG or aseg is **UNVERIFIED** (the README credits both tools). No manual labels | same |
| Extras | WM surface meshes (TopoFit), 4.5 GB. Full 3D download 6.6 GB | same |
| License / DUA | Users must follow the OASIS Data Use Agreement. The current OASIS DUA forbids sharing OASIS data "in whole or in part" with third parties without written permission. The wording names OASIS-3/4 explicitly ([OASIS DUA](https://sites.wustl.edu/oasisbrains/home/access/)). Whether derived OASIS-1 data may be redistributed is **UNVERIFIED**, so don't redistribute it. Release code and weights only | same |
| Within-session repeats | Raw OASIS-1 has 3–4 T1 MPRAGEs per session, plus a 20-subject reliability set rescanned within 90 days ([OASIS-1](https://sites.wustl.edu/oasisbrains/oasis-1/)). Neurite-OASIS exposes only one `orig` per subject. It is likely the FreeSurfer motion-corrected average (**UNVERIFIED**), so you need the raw OASIS-1 for the repeats | — |

Implications for your pipeline:
- Start from scanner-space `norm.nii.gz` + `seg35.nii.gz`. Register to MNI152 1 mm with ANTs, resampling labels with nearest-neighbour. Don't use `aligned_*`, which is in MNI305 space and cropped.
- Apply one per-image intensity normalization (for example a 99.5th-percentile scale) to **all** real datasets, so Neurite-OASIS doesn't form a separate intensity cluster. Synthetic-image training makes the model robust to this anyway.
- **Leakage:** Mindboggle-101's OASIS-TRT-20 group and the MICCAI-2012/Neuromorphometrics images both come from OASIS ([Mindboggle OSF](https://osf.io/yhkde/wiki), [MALF templates](https://github.com/muschellij2/malf.templates)). Remove those OASIS IDs from Neurite-OASIS training if either set is a test set. Whether these are exact OASIS-1 subject IDs is **UNVERIFIED**; cross-check the IDs.

### B. UltraCortex (OpenNeuro ds005216)

| Item | Finding | Source |
|---|---|---|
| ID / license | OpenNeuro **ds005216**, **CC0** | [arXiv 2406.18571](https://arxiv.org/html/2406.18571v1) |
| Size | 86 T1w images from 78 healthy adults (ages 20–53) | same |
| Repeats | 6 subjects scanned in 2 sessions and 1 subject in 3 sessions | same |
| Field / resolution | **9.4 T**, 0.6 / 0.7 / 0.8 mm isotropic, MPRAGE and MP2RAGE | same; [ultracortex.org](https://www.ultracortex.org/) |
| Contrasts | T1w only. Whether MP2RAGE INV1/INV2 images are included is **UNVERIFIED** | same |
| Labels | 12 brains with **manual GM/WM segmentations** (FreeSurfer initialization, then manual correction, validated by 2 neuroradiologists). Also SynthStrip masks, skull-stripped images and FreeSurfer ribbon files | same |
| Usefulness | (a) OOD 9.4 T, sub-mm test after resampling to 1 mm; (b) a small real within-subject rigid test (7 subjects with repeats); (c) a coarse 2–4 label manual deformable test (12 subjects). Not a training set (no subcortical labels, odd contrast). LUMIR25 used "high-field MRI" as a zero-shot domain; whether that was UltraCortex is **UNVERIFIED** | — |

## 3. Dataset comparison (C)

Role legend: **1** bulk T1 training · **2** real multi-contrast · **3** rigid/longitudinal/test-retest · **4** deformable eval with MANUAL labels · **5** atlas/groupwise eval · **6** pathology.

| Dataset | Access / turnaround | License / derivatives | N subj / sessions | Contrasts | Field / res | Longitudinal / repeats | Labels | Preproc avail. | Population | Roles |
|---|---|---|---|---|---|---|---|---|---|---|
| **IXI** | Direct download ([site](https://brain-development.org/ixi-dataset/)) | CC BY-SA 3.0 | ~600 healthy, 1 session | T1, T2, PD, MRA, DTI | 3 sites: HH Philips 3T, Guy's Philips 1.5T, IOP GE 1.5T | None | None (baselines used SynthSeg) | Preprocessed TransMorph-IXI exists (576 vols, CC BY-SA) ([repo](https://github.com/junyuchen245/Preprocessed_IXI_Dataset/blob/main/README.md)) | Healthy adults | **2, 5 (test only)** |
| **OASIS-1** (raw) | OASIS request form (all OASIS sets) ([request](https://sites.wustl.edu/oasisbrains/request-access/)). Institutional email required; turnaround **UNVERIFIED** | OASIS DUA: no redistribution, non-commercial | 416, plus 20 rescanned within 90 days | T1 (3–4 MPRAGE per session) | 1.5T ([OASIS-1](https://sites.wustl.edu/oasisbrains/oasis-1/)) | **Same-session repeats; 20-subject retest** | None (Neurite version: automated) | Neurite-OASIS (you have it) | 18–96, 100 mild–moderate AD | 1, 3 |
| **OASIS-2** | Same form | OASIS DUA | 150 / 373 sessions | T1 (3–4 per session) | 1.5T, single scanner ([OASIS-2](https://sites.wustl.edu/oasisbrains/oasis-2/)) | ≥2 visits, ≥1 yr apart, plus same-session repeats | None | — | 60–96, AD/nondemented | **3** (BrainMorph protocol) |
| **OASIS-3** | Same form; NITRC-IR account | OASIS DUA explicitly bans sharing | 1378 participants ([OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/)) | T1, T2, FLAIR (+PET) | 1.5/3T (**UNVERIFIED** detail) | Longitudinal over years | FreeSurfer (automated) | FreeSurfer outputs | 42–95, 622 with cognitive decline | 1, 2, 3 (MultiMorph used 1043 subj) |
| **ADNI** | LONI IDA application; DPC usually reviews **within ~2 weeks** ([ADNI](https://adni.loni.usc.edu/data-samples/adni-data/)) | DUA, no redistribution | Thousands, multi-visit | T1, T2/PD (ADNI1 dual-echo), FLAIR (later phases) ([ADNI MRI](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/)) | 1.5T and 3T | Yes, 6–12-month visits. 2501.13010 used 4 scans over 2 yrs | Automated | FreeSurfer outputs on IDA | MCI/AD/CN | **3**, 6 (atrophy) |
| **MIRIAD** | DUA on UCL DRC site ([NITRC](https://www.nitrc.org/projects/miriad), [FS list](https://mail.nmr.mgh.harvard.edu/archives/list/freesurfer@nmr.mgh.harvard.edu/message/MODF3WFE4TNQYWISAKEQ3NIRNRPDQEAN/)); turnaround **UNVERIFIED** | DUA | 69 (46 AD, 23 CN) / 708 scans | T1 | 1.5T, same scanner and radiographer ([PubMed](https://pubmed.ncbi.nlm.nih.gov/23274184/)) | Intervals of 2, 6, 14, 26, 38, 52 wks, 18, 24 mo; **same-day repeats** at some visits (**UNVERIFIED** which) | None | — | AD vs CN | **3** (2501.13010 protocol) |
| **HCP-YA** (incl. retest) | ConnectomeDB account plus open-access terms, click-through ([HCP-YA](https://w.humanconnectome.org/study/hcp-young-adult)) | Open-access DUT; restricted data separate | 1113 (1071 processed); **45 retest** ([S1200](http://www.humanconnectome.org/documentation/S1200/)) | T1w, T2w (0.7 mm) | 3T Connectome Skyra | Retest months apart (**UNVERIFIED** interval) | FreeSurfer (automated) | Minimally preprocessed, co-registered T1w/T2w | Healthy 22–35, twins | 1, 2, 3 |
| **HCP-Aging / Development** | **NDA** DUC, needs NIH-recognized institution and signing official ([NDA](https://nda.nih.gov/nda/access-data-info.html)) | NDA DUC | ~1200 / ~650 (**UNVERIFIED**) | T1w, T2w | 3T | No | FreeSurfer | Yes | Lifespan | Skip (slow) |
| **UK Biobank** | Application plus fee: **£1,000 for 3 yrs** (students £500) ([fees](https://ukbiobank.ac.uk/use-our-data/fees)) | Restrictive | ~50k+ imaging (**UNVERIFIED** current) | T1, T2-FLAIR, SWI, dMRI | 3T Skyra | Repeat imaging subset | Automated (FSL) | Yes | Population | Skip (paid, slow) |
| **ABCD** | NBDC DUC, institutional signing official ([ABCD](https://docs.abcdstudy.org/v/6_0_0/usage/access.html)) | DUC | ~12k youth | T1, T2 | 3T | Biennial | Automated | Yes | Children | Skip |
| **OpenBHB** | IEEE DataPort / site (**UNVERIFIED** route) | **UNVERIFIED** | ~5330 scans, 71 sites ([summary](https://www.emergentmind.com/topics/open-brain-health-benchmark-openbhb)) | T1 quasi-raw (affine-aligned) | Mixed | No | None | Linear-aligned, skull-stripped | Healthy; **includes IXI, ABIDE, CoRR, GSP, MPI-Leipzig** ([list](https://github.com/mansoorbalouch/Deep-Feature-Ensemble-Based-BrainAGE)) | Do not train on it (IXI leak) |
| **LUMIR 2024** | Google Drive (~52 GB) ([repo](https://github.com/JHU-MedImage-Reg/LUMIR_L2R)) | Inherits OpenBHB terms | >4000 T1 | T1 | 1 mm, 160×224×192, [0,255] | No | **Private** (evaluated on grand-challenge, with manual landmarks) | Skull-stripped, affine (OpenBHB) + N4 | Healthy | Optional pairwise benchmark; leaderboard: VFA 0.773 Dice val; test winner honkamj 0.785 Dice ([repo](https://github.com/JHU-MedImage-Reg/LUMIR_L2R)) |
| **LUMIR25** | Drive (~51 GB train, 430 MB val) ([repo](https://github.com/JHU-MedImage-Reg/LUMIR25_L2R25)) | Same | >5000 T1 train; val/test add high-field, pathology, other contrasts ([arXiv 2602.06292](https://arxiv.org/abs/2602.06292v1)) | T1 train; multi-contrast test | Mixed | No | Val: 10 coarse classes public; test: >100 classes, private | Yes | Mixed | Optional zero-shot contrast benchmark. Note the critique of its zero-shot claims ([LUMirage](https://arxiv.org/abs/2512.15505v1)) |
| **Kirby21** | NITRC, registered users ([NITRC](http://www.nitrc.org/projects/multimodal/)) | **UNVERIFIED** | 21 / 42 sessions | MPRAGE, FLAIR, T2 (plus qT1/qT2, MT, DTI) | 3T Philips | **Scan–rescan within hours** ([Frontiers](http://journal.frontiersin.org/article/10.3389/fnins.2016.00503/full)) | None | — | Healthy | 2, **3** |
| **CoRR** | INDI, open ([CoRR](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/)) | INDI terms (**UNVERIFIED** CC BY-NC) | ~1600 / ~5000 sessions (**UNVERIFIED**) | T1 (+fMRI/DTI) | 18 sites | Test–retest | None | — | Healthy | 3 (bulk test–retest) |
| **Maclaren test–retest** | Open (INDI/FCP, **UNVERIFIED** host) ([Sci Data](https://www.nature.com/articles/sdata201437)) | **UNVERIFIED** | 3 subjects, ~40 scans each (**UNVERIFIED** counts) | T1 MPRAGE | 3T | **Dense same-day + multi-day repeats** | None | — | Healthy | **3** (dense rigid) |
| **SIMON** | CONP/INDI, open ([GitHub](https://github.com/conpdatasets/SIMON-dataset)) | **UNVERIFIED** | 1 subject, 73 sessions, many sites | T1 (+others) | Many scanners | 15+ yrs, multi-site | None | — | Healthy | 3 (cross-scanner rigid, large N group) |
| **ON-Harmony** | OpenNeuro **ds004712**, open ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11992253/)) | OpenNeuro (CC0 typical; **UNVERIFIED**) | 20 traveling subjects × 6 scanners (165 sessions); 9 with 5 within-scanner repeats | T1w, T2w, 3D FLAIR, SWI | 3T, 3 vendors | **Traveling + within-scanner repeats** | None | Defaced BIDS | Healthy | **2, 3** (best open traveling-subject set) |
| **SRPBS traveling** | Application form to ATR ([ATR](https://bicr-resource.atr.jp/)) | ATR DUA | 9 subjects, 143 sessions, 12 sites ([news](https://medicalxpress.com/news/2021-08-multi-site-multi-disorder-resting-state-magnetic-resonance.html)) | T1 + rs-fMRI | 3T | Traveling | None | — | Healthy | Skip (ON-Harmony is open) |
| **Mindboggle-101** | OSF, direct ([OSF](https://osf.io/yhkde/wiki)) | Creative Commons (variant **UNVERIFIED**) | 101 | T1 | Mixed (OASIS-TRT, NKI, MMRR, HLN…) | OASIS-TRT-20 and NKI-TRT-20 are test–retest subsets | **Manual-edited DKT cortical labels**; manual subcortical on 20 ([Frontiers](https://www.frontiersin.org/articles/10.3389/fnins.2012.00171/pdf)) | Images in native and MNI152-affine space | Healthy | **4**, 5 |
| **LPBA40** | LONI download ([methods](https://resource.loni.usc.edu/wp-content/uploads/2017/08/Atlas-methods-LPBA40.pdf)) | LONI terms (**UNVERIFIED**) | 40 | T1 | 1.5T | No | **Manual, 56 structures** (58 delineated) | Skull-stripped versions | Healthy | **4** |
| **MICCAI 2012 / Neuromorphometrics** | Via challenge / MALF templates ([repo](https://github.com/muschellij2/malf.templates)) | **CC BY-NC** | 35 (OASIS) | T1 | 1.5T | Some OASIS repeats (**UNVERIFIED**) | **Manual, ~134 labels** (**UNVERIFIED** count) | — | Mixed | 4 (check OASIS overlap) |
| **IBSR18** | NITRC (**UNVERIFIED**) | **UNVERIFIED** | 18 | T1 | 1.5 mm coronal (**UNVERIFIED**) | No | Manual (**UNVERIFIED** count) | — | Healthy | 4 (low priority) |
| **CANDI** | NITRC (**UNVERIFIED**) | **UNVERIFIED** | ~100 children (**UNVERIFIED**) | T1 | 1.5T | No | Manual subcortical (**UNVERIFIED**) | — | Child psychiatric | 4, 6 (low priority) |
| **BraTS 2021+** | Synapse registration (**UNVERIFIED** turnaround, usually immediate) | Challenge terms (**UNVERIFIED** CC BY) | ~2000 patients ([summary](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/BraTS2021.md)) | T1, T1Gd, T2, FLAIR | Mixed, resampled to 1 mm, co-registered to SRI24, skull-stripped | No | Manual tumour labels | Fully preprocessed | Glioma | **2, 6**. Contrasts are **registered**, not intrinsically aligned |
| **BraTS-Reg** | Challenge site ([BraTS-Reg](https://bratsreg.github.io/)) | Challenge terms | ~250 pairs (**UNVERIFIED**) | T1, T1Gd, T2, FLAIR | 1 mm | Pre-op vs follow-up | Manual landmarks | Preprocessed | Glioma | 6 (longitudinal with pathology) |
| **QIN-GBM** | TCIA (open, **UNVERIFIED** license) | TCIA | Small (**UNVERIFIED**) | T1pre/post, T2, FLAIR, 5 mm stacks | Clinical | 2 sessions 2–5 days apart, pre/post surgery ([2501.13010](https://arxiv.org/html/2501.13010)) | None | — | GBM | **3, 6** (2501.13010 cross-contrast protocol) |
| **ATLAS v2.0** | Form → encryption key ([INDI](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html)) or ICPSR DUA ([ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/36684)) | DUA | 1271 (655 public train) ([paper](https://www.nature.com/articles/s41597-022-01401-7.pdf)) | T1 | Mixed | No | Manual lesion masks | MNI-registered | Stroke | 6 |
| **ABIDE I/II** | INDI, open | INDI terms (**UNVERIFIED**) | ~2200 (**UNVERIFIED**) | T1 | 20+ sites | No | None | — | Autism + TD | 1 (extra T1 diversity) |
| **Cam-CAN** | Online DUA for Stages I/II ([summary](https://atlaslongitudinaldatasets.ac.uk/datasets/cam-can)); turnaround **UNVERIFIED** (typically days) | DUA, no redistribution | ~650–700, ages 18–87 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/26375206/)) | T1 MPRAGE, T2 SPACE (**UNVERIFIED** detail) | 3T Siemens, 1 mm | No | None | — | Healthy lifespan | 1, **2** |
| **MPI-Leipzig MBB / LEMON** | INDI/NITRC, open ([LEMON](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html), [MBB](http://fcon_1000.projects.nitrc.org/indi/retro/MPI.html)) | **UNVERIFIED** (open, CC-type) | 321 (MBB), 228 (LEMON) | **MP2RAGE (INV1/INV2/UNI)**, T2, FLAIR | 3T Siemens | No | None | — | Healthy | **2** (MultiMorph trained on MBB T2/FLAIR) |

Key structural facts:
- **Intrinsically co-registered contrasts** come from the same acquisition, so they have an exact zero-motion relation. Examples: dual-echo PD/T2 (ADNI-1 used dual-echo T2 per [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/)) and MP2RAGE INV1/INV2/UNI. These are the only real-data source of "exact" multi-contrast rigid ground truth. Whether IXI PD/T2 are dual-echo (one acquisition) is **UNVERIFIED**; check that the NIfTI headers and grids match. KeyMorph and BrainMorph describe IXI T1/T2/PD as "in spatial alignment" ([KeyMorph](https://arxiv.org/html/2304.09941v2)), but IXI T1 is a separate acquisition, so treat T1↔T2 as only approximately aligned.
- BraTS and HCP contrasts are **registered** after acquisition, so their ground truth carries registration error.

## 4. Baseline datasets and protocols (D)

| Baseline | Training data | Evaluation data / protocol | Metrics | Weights public | Splits public |
|---|---|---|---|---|---|
| **MultiMorph** (CVPR 2025) | OASIS-1 (416), OASIS-3 (1043 subj, T1/T2), MPI MBB (T2, FLAIR), 50% synthetic groups; group size m ∈ [2,12]. Preproc: ANTs affine to a 1 mm reference, SynthStrip, SynthSeg (30 labels) ([arXiv](https://arxiv.org/html/2504.00247v1)) | Held-out 20% of training sets. **Unseen IXI, Guy's site, 319 subjects, T1/T2/PD**. Subgroup atlases of size 5–60 vs ANTs SyGN, AtlasMorph, Aladdin | Dice (segmentation transfer), folds, centrality, construction time | Yes, `models/model_cvpr.pth` ([GitHub](https://github.com/mabulnaga/multimorph)) | No (random 20% hold-out); training code "coming soon" |
| **BrainMorph** | >100k volumes, ~16k subjects, public datasets (list in appendix); HD-BET, SynthSeg (33 regions); 256³ 1 mm ([arXiv](https://arxiv.org/html/2405.14019v1)) | **IXI 100 subjects** (uni/multi-modal T1/T2/PD), **BraTS test split**, an MS dataset (60 pts), **OASIS-2 longitudinal** (rigid groupwise over all timepoints); cross-subject groupwise N ∈ {4…128}; rotations applied on all 3 axes | Dice, runtime | Yes ([GitHub](https://github.com/alanqrwang/brainmorph)) | **UNVERIFIED** |
| **KeyMorph** | IXI 427/50/100 split, skull-stripped, SynthSeg 23 ROIs ([arXiv](https://arxiv.org/html/2304.09941v2)) | IXI 100 test subjects, all modality pairings, rotation sweeps | Dice, HD, Jacobian stats | Yes (**UNVERIFIED** for this version) | **UNVERIFIED** |
| **SynthMorph** (affine/joint, Hoffmann et al.) | Synthetic only, from 100 label maps (FSM, OASIS, …) ([arXiv](https://arxiv.org/html/2301.11329v1)) | GSP, IXI (T1/T2/PD), UKBB, HCP-D, MASiVar, QIN; skull-stripped copies of GSP/IXI/UKBB; SynthSeg labels; pairs such as GSP→IXI-T1 | Dice (GM/WM/CSF for affine), affine residual | Yes (FreeSurfer `mri_synthmorph`; [synthmorph.io](https://synthmorph.voxelmorph.net/)) | No (**UNVERIFIED**) |
| **arXiv 2501.13010** (longitudinal rigid) | Synthetic from SynthMorph's 100 label maps ([arXiv](https://arxiv.org/html/2501.13010)) | **ADNI** 1.5T and 3T (4 scans / 2 yrs) and **MIRIAD** (7 scans, 2 wk–2 yr); 5 subjects each for validation, one random cross-time pair per test subject, with and without skull-strip; **QIN** cross-contrast (T1pre–FLAIR, T1pre–T1post, FLAIR–T2). Conformed 256³ LIA; SynthSeg | Dice over 5 tissue classes (L/R cortex, subcortex, cerebellum) | Probably via SynthMorph tool (**UNVERIFIED**) | No |

Other baselines: ANTs and `mri_robust_template` need no data. SynthMorph and LUMIR leaderboards give ready-made numbers for SynthMorph, uniGradICON, BrainMorph and SyN on LUMIR validation ([LUMIR repo](https://github.com/JHU-MedImage-Reg/LUMIR_L2R)).

No baseline publishes subject-level test IDs that I could confirm. For direct comparisons, **re-run the public baseline weights on your own fixed splits**. Don't copy their numbers.

## 5. Recommended plan

### Access requests, ordered by priority and approval time

1. **OASIS form, today.** One request covers OASIS-1/2/3 ([request](https://sites.wustl.edu/oasisbrains/request-access/)). Write a complete research statement; incomplete ones are denied. You need the raw OASIS-1 repeats, OASIS-2 for BrainMorph's longitudinal protocol, and OASIS-3 T2/FLAIR as real multi-contrast training data matching MultiMorph.
2. **ADNI, today.** About 2 weeks ([ADNI](https://adni.loni.usc.edu/data-samples/adni-data/)). Needed for the 2501.13010 protocol and for dual-echo PD/T2.
3. **MIRIAD, today.** Online DUA. Needed for the 2501.13010 protocol and same-day repeats.
4. **Cam-CAN, this week.** Online DUA. Extra T1/T2 training diversity. Optional.
5. No approval needed: IXI, ON-Harmony (ds004712), Kirby21, Mindboggle-101, LPBA40, MPI-Leipzig MBB/LEMON, QIN (TCIA), BraTS (Synapse, fast).
6. **Skip:** UK Biobank (fee, slow), ABCD and HCP-A/D (NDA plus institutional signatures), SRPBS (form-based, and ON-Harmony covers it), OpenBHB/LUMIR as training data (IXI and MPI-Leipzig leak).

### Splits per role

| Role | Train | Val | Test |
|---|---|---|---|
| 1 Bulk T1 | Neurite-OASIS (minus Mindboggle/MICCAI-overlapping IDs), OASIS-3 T1, optional Cam-CAN/ABIDE; plus synthetic from SynthSeg labels | 5% hold-out from each | — |
| 2 Real multi-contrast | OASIS-3 T2/FLAIR, MBB T2/FLAIR (as MultiMorph did), optional Cam-CAN T2 | hold-out | IXI T1/T2/PD (PD never seen in training), Kirby21, ON-Harmony |
| 3 Rigid / longitudinal | Supervised synthetic transforms only (no real pairs needed) | 5 ADNI + 5 MIRIAD subjects (matches 2501.13010) | ADNI + MIRIAD (2501.13010 protocol), **OASIS-2** (BrainMorph groupwise-longitudinal protocol), ON-Harmony and SIMON (cross-scanner, large-N groups), UltraCortex repeats (OOD) |
| 3b Exact-GT multi-contrast rigid | — | — | Intrinsically co-registered pairs (MP2RAGE INV1/INV2/UNI from LEMON/MBB; dual-echo PD/T2 from ADNI-1 or IXI once confirmed) **plus known random rigid transforms**, so ground truth = applied transform |
| 4 Deformable, manual labels | — | — | **Mindboggle-101** (DKT cortex + 20 manual subcortical), **LPBA40** (56 manual), UltraCortex 12 (GM/WM) |
| 5 Atlas / groupwise | — | — | **IXI-Guy's 319, T1/T2/PD** (exact MultiMorph protocol: Dice transfer, folds, centrality, time); subgroups 5–60 |
| 6 Pathology | — | — | BraTS (affine only), ATLAS (optional), QIN |

### How the plan maps to the experiments

- **Longitudinal rigid vs `mri_robust_template` / ANTs / SynthMorph / BrainMorph.**
  - Use ADNI + MIRIAD with the 2501.13010 Dice protocol so those numbers can be compared.
  - Add groupwise metrics: halfway-space **bias** (‖mean log-transform‖ of predicted transforms) and **cycle consistency** (‖T_ij ∘ T_jk ∘ T_ki − I‖) over all timepoints. MIRIAD (up to 12 scans per subject) and OASIS-2 (2–5 visits) give natural N > 2 groups.
- **Composed vs direct pairwise accuracy.** Same subjects. Compare T_i→c ∘ T_c→j with a direct pairwise run, using Dice and transform RMS. IXI for cross-subject, MIRIAD for within-subject.
- **Runtime vs N.** ON-Harmony (6 scanners × 20 subjects), SIMON (73 sessions of one subject) and IXI (N up to 319) give within- and cross-subject groups of growing size with no labels needed.
- **Atlas building vs MultiMorph / ANTs.** IXI-Guy's with the published metric set. The released MultiMorph checkpoint lets you rerun it.

## 6. Open risks and unknowns

- **Turnaround** for OASIS, MIRIAD and Cam-CAN is not published (**UNVERIFIED**). ADNI's ~2 weeks is the only documented figure. Ask for all of them now.
- **IXI PD/T2 dual-echo and MP2RAGE inversion availability** (in LEMON/MBB/UltraCortex) need on-disk checks before you claim "exact" multi-contrast ground truth.
- **OASIS ID overlap** between Neurite-OASIS and Mindboggle OASIS-TRT-20 / MICCAI-2012 must be checked by subject ID.
- **License wording.** Mindboggle (CC variant), Kirby21, CoRR, ON-Harmony (OpenNeuro CC0?) and BraTS need confirming before any derived release. Assume no redistribution of OASIS/ADNI/MIRIAD derivatives.
- **Baseline training contamination.** BrainMorph's 100k training list may include OASIS or ADNI subjects (the full list is in its appendix, which I didn't review). Check it before claiming held-out comparisons on OASIS-2.
- **MIDL 2027 deadline** is not posted yet. The ~3.5-month window is an assumption.
- **The MIRIAD same-day repeat schedule** (which visits) is **UNVERIFIED**.
