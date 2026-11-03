# ParaNAM Trainer 0.5.4

- Static and parametric training now use the official NAM v3 training interval:
  skip the first 10 seconds and the final 9 seconds. This applies to Standard
  and TTS input v11; custom inputs must retain the same endpoint structure.
- Every completed epoch reports **Validation ESR** and **Train ESR**, separately
  for Full and Lite. Validation ESR retains the separate validation captures.
  Train ESR evaluates only the final nine seconds of the training captures,
  with frozen end-of-epoch weights and no gradient updates. It uses NAM's ESR
  function, whole-clip evaluation and receptive-field/delay handling, matching
  the official static trainer's validation measurement. Multiple training
  captures contribute equally to the mean. Train ESR is not the minibatch loss
  accumulated while learning.
- Queue has independent **Best Validation ESR (Full)** and **Best Train ESR
  (Full)** cards. The minima may occur at different epochs. Best checkpoint
  selection and intermediate/final exports still use Validation ESR only.
- Queue now has **Full ESR chart**, **Lite ESR chart** and **Training log** tabs.
  Each size chart shows Validation ESR in blue and Train ESR in dashed orange.
  Both scores, best Train ESR and the chart history survive pause/resume.
- Added space between navigation icons and labels, and renamed **Projects**
  to **Project**.

**Start a new training instance to use the new data split.** Older jobs cannot
resume into this protocol, since their weights may already have learned the
validation segments. Existing captures and exported models are retained;
use Retrain to reuse the recordings.

Train ESR adds an inference pass over each training capture's nine-second tail
at every epoch. Epoch duration therefore depends on the number of captures.

[Training controls and measurement details](https://github.com/DLC86/ParaNAM-Trainer/blob/main/docs/training-controls.md).

Packages: Windows x64 Setup, Linux x86-64 DEB and portable archive, macOS Apple
Silicon and Intel DMGs. Each native build verifies training, pause/resume,
the persistent queue and Full/Lite playback before upload.
