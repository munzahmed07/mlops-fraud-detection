import os
import logging
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

REFERENCE_DATA_PATH = "data/features/X_train.csv"
CURRENT_DATA_PATH = "logs/predictions.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.txt")


def main():
    logging.info("===== DRIFT CHECK STARTED =====")
    os.makedirs(REPORT_DIR, exist_ok=True)

    logging.info("Loading reference (training) data...")
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)

    logging.info("Loading current (production) data...")
    cur_df = pd.read_csv(CURRENT_DATA_PATH)

    # create comparable feature
    ref_sample = ref_df.iloc[:, :1].copy()
    ref_sample.columns = ["score_feature"]

    cur_sample = cur_df[["fraud_probability"]].copy()
    cur_sample.columns = ["score_feature"]

    # -----------------------------
    # Run Evidently
    # -----------------------------
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_sample, current_data=cur_sample)

    # -----------------------------
    # Extract drift result safely
    # -----------------------------
    drift_detected = "UNKNOWN"

    try:
        # works in very old versions
        metrics = report._metrics
        for m in metrics:
            if hasattr(m, "drift_detected"):
                drift_detected = m.drift_detected
    except Exception:
        pass

    # -----------------------------
    # Save simple report
    # -----------------------------
    with open(REPORT_PATH, "w") as f:
        f.write("DRIFT REPORT\n")
        f.write("====================\n")
        f.write(f"Drift detected: {drift_detected}\n")
        f.write("Feature checked: score_feature\n")
        f.write("Method: Evidently DataDriftPreset\n")

    logging.info(f"Drift report saved to: {REPORT_PATH}")
    logging.info(f"Drift detected: {drift_detected}")
    logging.info("===== DRIFT CHECK COMPLETED =====")


if __name__ == "__main__":
    main()
