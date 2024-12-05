from .experiment import Experiment
from .ingestion import HuggingFaceDataset
from .model import HuggingFaceModel
from .register import Register

title = "summarization"
key_metric = "rougeLsum/v1/p90"

if __name__ == "__main__":
    data = HuggingFaceDataset()
    test_df = data.load_pd_test_dataset(path="billsum", split="ca_test")
    models = {
        "default_hugging_face": HuggingFaceModel(),
        "google_pegasus-xsum": HuggingFaceModel(model_name="google/pegasus-xsum"),
    }
    results = []
    for run_name, model in models.items():
        exp = Experiment(model=model, title=title)
        model_info = exp.track(run_name=run_name)
        metrics = exp.evaluate(model_info.model_uri, test_df=test_df)
        results.append(metrics[key_metric])
    best_run_name = list(models)[results.index(min(results))]
    reg = Register(title=title)
    run_id = exp.search_finished_experiments(run_name=best_run_name, max_results=1)["run_id"][
        0
    ]
    reg.register_model(run_id=run_id)
