# Third-party notices

## Yuaz SGR
Yuaz DDSP Resampler requires Yuaz SGR at runtime. Yuaz SGR source/checkpoints are not redistributed by this package; their own terms apply.

## OpenUtau
OpenUtau is not bundled. Yuaz uses its external-resampler integration.

## GTSinger developer training data
GTSinger is optional developer training data and is licensed by its authors under CC BY-NC-SA 4.0. GTSinger audio is not bundled by this source package. Preserve the dataset attribution/license information with any redistributed derived control weight.

## VocalSet developer training data
Canonical source: Julia Wilkins, Prem Seetharaman, Alison Wahl, and Bryan Pardo, *VocalSet: A Singing Voice Dataset*, Zenodo record 1442513. The developer setup uses the `Bill13579/vocalset-mirror` Hugging Face repository and can reach it through official Hugging Face or `hf-mirror.com`. The mirror identifies the dataset as CC BY 4.0 and as a mirror of the Zenodo release. VocalSet audio/parquet data is not bundled in Yuaz.

## OSF Phonation Modes developer training data
The YT Tension developer pipeline uses the public Phonation Modes Dataset in OSF project `pa3ha`, file GUID `cwquj` (`ND357A_24bit_cut_ALL.zip`). The dataset contains sustained singing examples labelled breathy, neutral/modal, flow and pressed. Yuaz uses breathy, neutral/modal and pressed for the signed Tension axis; flow is retained in the downloaded dataset but is not used as a direct endpoint in v1. Raw recordings are not redistributed by this source package. Preserve the source project attribution/license metadata with any derived-weight release.

## MOCHA-TIMIT developer training data
MOCHA-TIMIT is downloaded from the official Centre for Speech Technology Research (CSTR) distribution. The public corpus README/archives permit research, educational and individual non-commercial use and contain the corpus licensing information. Yuaz downloads only the two public speaker archives for developer training. The raw microphone, laryngograph, EMA and label data are not bundled with Yuaz.
