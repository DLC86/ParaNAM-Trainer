# ParaNAM Trainer 0.8.2

Retrain now preserves the capture plan of the selected job when opening projects made with older trainer versions. Previously, a legacy Avoid limit could disappear during migration. Accepting the resulting plan update could replace recorded captures at the lowest audible setting with synthetic Mute targets. The fix also applies when retraining a job that was itself created with Retrain.

To repeat an earlier training run with its original captures, select the queue entry that still contains those recordings and choose Retrain. Existing jobs keep their saved datasets.

Resume now recognizes a frozen job even when that job was originally created through Retrain. If 0.8.1 already converted its old Avoid limits into equivalent rules during a failed Resume, 0.8.2 can verify the original checkpoint using the saved control settings and the current recordings. A mismatch still blocks Resume.
