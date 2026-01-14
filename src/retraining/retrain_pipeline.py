import os
import logging
import subprocess

DRIFT_REPORT_PATH = "reports/drift_report.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def drift_detected():
    """
    Reads drift report and decides if retraining is needed.
    """
    if not os.path.exists(DRIFT_REPORT_PATH):
        logging.warning("Drift report not found. Skipping retraining.")
        return False

    with open(DRIFT_REPORT_PATH, "r") as f:
        content = f.read().lower()

    if "true" in content:
        return True

    # if unknown, still retrain (safe policy)
    if "unknown" in content:
        logging.info("Drift status unknown → retraining as safe default.")
        return True

    return False


def run_command(cmd):
    logging.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    logging.info("===== RETRAINING PIPELINE STARTED =====")

    if not drift_detected():
        logging.info("No drift detected. Skipping retraining.")
        return

    logging.info("Drift detected → starting retraining pipeline")

    # -----------------------------
    # 1. Feature Engineering
    # -----------------------------
    run_command("python src/features/build_features.py")

    # -----------------------------
    # 2. Train models with MLflow
    # -----------------------------
    run_command("python src/training/train_with_mlflow.py")

    # -----------------------------
    # 3. Register best model
    # -----------------------------
    run_command("python src/training/register_model.py")

    logging.info("===== RETRAINING PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()
