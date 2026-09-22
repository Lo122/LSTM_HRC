set -e
PY=hrc_communication/.venv/Scripts/python.exe
mkdir -p bench/results bench/logs
# All arms: peak targets + mirror, 15-fold LOSO, fold weights saved for the
# event-level trigger evaluation.
# R0 is the control: 7 classes AND uncapped, i.e. both old settings, so the
# Lift-drop and the cap can be attributed separately.
$PY bench/loso.py --targets peak --aug mirror --keep-lift --pos-weight-cap 0 \
    --save-models --tag R0_7class > bench/logs/R0_7class.log 2>&1
$PY bench/loso.py --targets peak --aug mirror --pos-weight-cap 0 \
    --save-models --tag R1_6class_nocap > bench/logs/R1_6class_nocap.log 2>&1
$PY bench/loso.py --targets peak --aug mirror --pos-weight-cap 2.0 \
    --save-models --tag R2_cap > bench/logs/R2_cap.log 2>&1
$PY bench/loso.py --targets peak --aug mirror --pos-weight-cap 2.0 --focal-gamma 2.0 \
    --save-models --tag R3_focal > bench/logs/R3_focal.log 2>&1
echo ALL_RETRAIN_ARMS_DONE
