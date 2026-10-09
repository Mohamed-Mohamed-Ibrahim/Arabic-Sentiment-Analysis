import mlflow
import torch
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Model Training 1"
METRIC_NAME = "test_accuracy"
TRACKING_URI = "http://localhost:5000/"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{METRIC_NAME} DESC"],
        max_results=1,
    )

    if not runs:
        print("No runs found in this experiment.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_score = best_run.data.metrics.get(METRIC_NAME, "N/A")

    print(f"Best run: {best_run_id} ({METRIC_NAME}: {best_score})")

    current_prod_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.status = 'production'",
    )
    for run in current_prod_runs:
        if run.info.run_id != best_run_id:
            client.set_tag(run_id=run.info.run_id, key="status", value="archived")

    client.set_tag(run_id=best_run_id, key="status", value="production")
    print("Promoted run to status='production'.")

    with open("best_run.txt", "w") as f:
        f.write(f"run_id: {best_run_id}\n{METRIC_NAME}: {best_score}\n")

    with open("best_run.txt", "r") as f:
        for line in f:
            if line.startswith("run_id:"):
                run_id = line.split(":", 1)[1].strip()
                break

    print(f"Loading run: {run_id}")

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    dummy_text = torch.tensor([1, 2, 3], dtype=torch.int64).to(device)
    dummy_offsets = torch.tensor([0], dtype=torch.int64).to(device)
    torch.onnx.export(
        model,
        (dummy_text, dummy_offsets),
        "notebooks/model.onnx",
        opset_version=17,
        input_names=["text", "offsets"],
        output_names=["output"],
        dynamic_axes={
            "text": {0: "total_tokens"},
            "offsets": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
