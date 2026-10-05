# EmbryoFusion AI

**Multimodal fusion of patient records and embryo images for IVF outcome prediction**

<p align="center">
  <img src="docs/images/example_cases.png" alt="Cases where the clinical record and the embryo image disagree" width="920">
</p>

## Project brief

Whether an embryo transfer leads to a pregnancy depends on two things that are recorded in two completely different places. The first is the patient: her age, her hormone profile, the thickness of the uterine lining on the day of transfer, how many transfers have already failed. That information lives in rows and columns in the clinic's electronic record. The second is the embryo itself: how far it has expanded, how many cells make up the inner cell mass that will become the fetus, how cohesive the outer trophectoderm layer is. That information lives in a microscope image on the embryology laboratory's imaging system, and it reaches the clinical record, if at all, as a three character grade typed in by hand.

Most predictive models in reproductive medicine use one of these sources and ignore the other. Tabular models built on registry data know the patient and treat every embryo of a given grade as identical. Image models built on embryo photographs can rank embryos but have no idea whether they are going into a receptive uterus. A clinician weighs both at once, and so should a model. The open questions are practical ones: how much does adding the second view really improve prediction, does the model use both views or quietly lean on one, what happens when a view is missing, and is a jointly trained network worth its complexity compared with simply combining two separate models?

EmbryoFusion AI is built to answer those questions under controlled conditions. It trains a dual branch neural network (a dense network for the clinical record and a convolutional or Vision Transformer encoder for the day 5 embryo image, concatenated before the classification head) and compares it with single modality baselines and with a late fusion stack, using paired bootstrap tests on identical patients. The project puts as much weight on the data layer as on the model, because multimodal work fails most often at the join: the repository includes schema validation, cleaning with an audit trail, modality pairing with orphan detection, patient level splitting and tabular preprocessing, each covered by unit tests that run in GitHub Actions on every push. Experiments are tracked with Weights and Biases and the image branch can use Hugging Face Transformers models.

Paired clinical and embryo image data is among the most restricted data in medicine and no public set links the two. The project therefore generates 10,000 paired transfers: a clinical record per patient and a rendered day 5 blastocyst image whose appearance is driven by Gardner grades. Embryo quality is never written into the clinical table, so the only way a model can learn it is from pixels. All results describe this simulated cohort and none is a clinical claim.

## What the system does

<table>
  <tr><th align="left">Capability</th><th align="left">How it is delivered</th></tr>
  <tr><td>Dual branch fusion network</td><td>Dense encoder for 15 preprocessed clinical features and an image encoder, concatenated into a shared head, with modality dropout</td></tr>
  <tr><td>Choice of image encoder</td><td>Compact residual CNN, or a Vision Transformer from Hugging Face Transformers (trained from a configuration or loaded from the Hub)</td></tr>
  <tr><td>Baselines</td><td>Clinical record only, embryo image only, and late fusion by stacking the two single modality models</td></tr>
  <tr><td>Statistical comparison</td><td>Bootstrap intervals for each model and paired bootstrap for differences between models on the same cases</td></tr>
  <tr><td>Reliance audit</td><td>Each view withheld or swapped with another patient's at inference to measure what the fusion model really uses</td></tr>
  <tr><td>Data processing layer</td><td>Validation, cleaning, pairing, patient level split and preprocessing as small pure functions with an audit of every correction</td></tr>
  <tr><td>CI and delivery</td><td>GitHub Actions: lint, data processing tests on three Python versions, one epoch smoke training, tagged release with model artifacts</td></tr>
  <tr><td>Experiment tracking</td><td>Weights and Biases runs (offline by default) mirrored to JSON so reports never depend on an external service</td></tr>
</table>

## Key findings

### Two views beat one, and the gain over the best single view is statistically real

On 1,441 transfers from patients held out of training, the clinical record alone reaches a ROC AUC of 0.647 and the embryo image alone reaches 0.719. The fusion network reaches 0.743, against a ceiling of 0.769 that the simulator's own true probabilities achieve.

<table>
  <tr><th align="left">Model</th><th>ROC AUC (95% interval)</th><th>PR AUC</th><th>Brier score</th><th>Parameters</th></tr>
  <tr><td>Clinical record only</td><td align="center">0.647 (0.617 to 0.675)</td><td align="center">0.469</td><td align="center">0.203</td><td align="center">5,473</td></tr>
  <tr><td>Embryo image only (CNN)</td><td align="center">0.719 (0.690 to 0.747)</td><td align="center">0.529</td><td align="center">0.189</td><td align="center">559,801</td></tr>
  <tr><td>Late fusion (stacked single models)</td><td align="center">0.743 (0.713 to 0.769)</td><td align="center">0.575</td><td align="center">0.181</td><td align="center">565,277</td></tr>
  <tr><td>Joint fusion, ViT image branch</td><td align="center">0.730 (0.703 to 0.757)</td><td align="center">0.565</td><td align="center">0.185</td><td align="center">337,985</td></tr>
  <tr><td><b>Joint fusion, CNN image branch</b></td><td align="center"><b>0.743 (0.713 to 0.768)</b></td><td align="center"><b>0.577</b></td><td align="center"><b>0.182</b></td><td align="center">565,145</td></tr>
</table>

<p align="center"><img src="docs/images/roc_curves.png" alt="ROC curves" width="470"></p>
<p align="center"><img src="docs/images/auc_intervals.png" alt="AUC with bootstrap intervals" width="640"></p>

The individual intervals overlap, which is why the project does not stop there. Scoring both models on the same resampled patients removes most of the shared noise: the fusion model beats the image only model by 0.024 AUC (95 percent interval 0.007 to 0.040, positive in 99.8 percent of resamples) and beats the clinical record model by 0.096 (0.068 to 0.124). In operational terms, transfers in the top quarter of fused scores ended in pregnancy 59 percent of the time and those in the bottom quarter 12 percent of the time, against a cohort rate of 30 percent.

### A jointly trained network did not beat a simple stack of two models

This is the result a less careful write up would leave out. Late fusion (train the two single modality models separately, then fit a two input logistic regression on their validation predictions) scores 0.743, the same as the jointly trained network. The paired difference is 0.0003 with an interval of minus 0.007 to plus 0.008. In this cohort the patient and embryo contributions combine almost additively, so there is little for a joint representation to discover.

The engineering conclusion is useful. Late fusion is easier to build, lets each team own its model, and should be the default comparator for any multimodal claim. Joint fusion earns its place for other reasons, shown in the next finding: one artifact to deploy, and graceful behaviour when a view is missing.

### The fusion model uses both views and degrades gracefully when one is missing

<p align="center"><img src="docs/images/modality_ablation.png" alt="Fusion model performance with each view withheld or swapped" width="640"></p>

With the image withheld, the fusion model scores 0.643, essentially the same as the dedicated clinical record model (0.647). With the record withheld it scores 0.714, close to the dedicated image model (0.719). Modality dropout during training is what makes this possible: one network can serve a clinic whose imaging system is offline without a separate fallback model.

The more important number is what happens when the image belongs to a different patient. Performance falls to 0.565, far below the 0.643 obtained with no image at all. **A mismatched image is worse than a missing one.** In a multimodal system, linkage errors between the laboratory imaging system and the clinical record are not noise that washes out; they actively mislead the model. That is the reason this repository treats pairing as a first class, tested component.

### Without pretraining, the Vision Transformer trails the CNN

The Hugging Face ViT branch, trained from scratch on 6,846 images, reaches 0.730 against 0.743 for the CNN branch. The paired interval for the difference (minus 0.005 to plus 0.030) includes zero, so the honest reading is that the transformer is no better and probably slightly worse at this data scale, which matches the general experience that transformers need pretraining or far more data. The configuration accepts a Hub model name so a pretrained ViT can be dropped in when internet access and larger images are available.

### Where each view matters

<p align="center"><img src="docs/images/subgroup_auc.png" alt="AUC by maternal age and embryo quality" width="860"></p>

Among patients aged 38 and over the image carries almost everything: the image model scores 0.773 and fusion adds nothing (0.774), because in that group embryo quality is the dominant source of variation. Under 35 the two views are complementary (0.640 and 0.706 alone, 0.740 fused). Within a fixed embryo quality group the picture reverses: once the grade is held constant the image has much less to say (0.60 to 0.62) and the clinical record matches or exceeds it (0.61 to 0.65), with fusion ahead in both groups (0.686 for good embryos, 0.627 for poorer ones). The practical message for counselling is that the patient's record matters most when choosing whether to transfer a given embryo, and the image matters most when choosing between embryos.

### The data layer found and fixed real problems before any model ran

<table>
  <tr><th align="left">Processing step</th><th>Count</th></tr>
  <tr><td>Duplicate transfer records removed</td><td align="center">40</td></tr>
  <tr><td>Diagnosis labels normalised</td><td align="center">422</td></tr>
  <tr><td>Impossible BMI values blanked</td><td align="center">31</td></tr>
  <tr><td>Clinical records with no embryo image</td><td align="center">150</td></tr>
  <tr><td>Embryo images with no clinical record</td><td align="center">100</td></tr>
  <tr><td>Transfers successfully paired</td><td align="center">9,750</td></tr>
</table>

Splitting is done by patient, so a woman with two transfers never appears in both training and test data, and a test asserts that the reference embryo grades used for analysis can never enter the model inputs.

<p align="center"><img src="docs/images/training_curves.png" alt="Validation AUC by epoch for each tracked run" width="560"></p>

## Architecture

```
clinical.csv                         image_manifest.csv + images/
     |                                         |
validate_schema, clean_clinical               |
     |                                         |
     +===========> pair_modalities <===========+        inner join, orphan audit, file existence check
                         |
                   patient_split                         train, val, test by patient
                         |
        +================+=================+
        |                                  |
tabular preprocessor                 image loading and per image standardisation
(impute, flag, scale, one hot)             |
        |                                  |
  TabularEncoder (dense)        CNNImageEncoder or ViTImageEncoder (Hugging Face)
        |                                  |
        +==========> concatenate <=========+             modality dropout during training
                         |
                 classification head ==> probability of clinical pregnancy
                         |
   metrics with bootstrap intervals, paired comparisons, reliance audit, Weights and Biases runs
```

## Repository layout

```
embryofusion_ai
    configs/config.yaml                    data, model, experiments, tracking
    artifacts/fusion_model.pt              selected fusion model
    artifacts/tabular_preprocessor.joblib  fitted preprocessing pipeline
    data/sample                            24 example transfers with images
    docs/images                            figures in this document
    reports/metrics.json                   all results, comparisons and the processing audit
    src/embryofusion
        data/render.py                     procedural blastocyst renderer
        data/simulate.py                   paired dataset generator with realistic faults
        data/processing.py                 validation, cleaning, pairing, splitting, preprocessing
        models/fusion.py                   tabular, CNN and ViT encoders and the fusion network
        training/pipeline.py               experiments, late fusion, ablations, export
        training/tracker.py                Weights and Biases wrapper with JSON mirror
        evaluation/metrics.py              bootstrap and paired bootstrap
        evaluation/report.py               figure generation
        inference/predict.py               scoring with one or both views
    tests
        test_data_processing.py            12 unit tests of the data layer (marker: data)
        test_models_and_pipeline.py        model variants and a full one epoch pipeline run
    .github/workflows/ci.yml               lint, data tests on three Python versions, smoke training
    .github/workflows/release.yml          tagged release with package and model artifacts
```

## Getting started

Python 3.10 or later is required.

```
make install
make all
make test
```

`make all` renders the paired dataset, trains the four experiments, runs the comparisons and rebuilds the figures. On a single CPU core it takes about fifteen minutes and it is much faster on a GPU. Trained artifacts are included, so prediction works straight after installation. To run only the fast data processing tests that continuous integration runs on every push, use `make datatest`.

### Scoring a transfer

```python
from embryofusion.inference.predict import FusionPredictor

predictor = FusionPredictor()
record = {"female_age": 33, "bmi": 23.0, "amh_ng_ml": 2.5, "endometrial_thickness_mm": 10.0,
          "previous_failed_transfers": 0, "infertility_duration_years": 2.0,
          "infertility_cause": "unexplained", "frozen_transfer": 1}

print(predictor.predict(record, "data/sample/images/T000001.png"))   # both views
print(predictor.predict(record))                                      # clinical record only
```

With both views the response includes the fused probability and what each view would have said alone. Fields missing from the record are imputed by the fitted preprocessing pipeline.

### Experiment tracking

Runs are written to `reports/tracking` in Weights and Biases offline format. After `wandb login`, upload them with the `sync` target in the Makefile, or set `tracking.mode` to `online` in the configuration to stream runs live. Set it to `disabled` to skip the dependency entirely; the JSON mirror keeps the report step working.

### Using real data and pretrained encoders

Place a `clinical.csv` and an `image_manifest.csv` in the layout shown in `data/sample`, point `data.root` at the folder and run the `train` target. To use a pretrained Vision Transformer, set `model.vit.pretrained_name` to a Hub model such as the ImageNet 21k ViT base checkpoint and raise `data.image_size` to 224.

## Limitations and responsible use

* The cohort is simulated and the images are rendered. Real embryo images vary with microscope, focal plane and laboratory, and real outcomes depend on factors not modelled here, including embryo ploidy.
* The outcome process in the simulator is close to additive, which favours late fusion. Real data may contain stronger interactions between patient and embryo.
* Images are 64 pixels square to keep CPU training practical. Fine trophectoderm detail is lost at that size.
* Predictions describe one transfer. They are not a cumulative success estimate.
* EmbryoFusion AI is a research and portfolio prototype. It is not a medical device.

## Licence

Released under the MIT licence.
