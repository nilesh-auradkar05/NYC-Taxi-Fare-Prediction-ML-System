import builtins
import importlib
import sys


def test_live_pipeline_modules_do_not_import_metaflow(monkeypatch):
    for module_name in [
        "metaflow",
        "src.common.pipeline",
        "src.pipelines.training",
        "src.pipelines.inference",
    ]:
        sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "metaflow" or name.startswith("metaflow."):
            raise AssertionError("live pipeline modules must not import Metaflow")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    training = importlib.import_module("src.pipelines.training")
    inference = importlib.import_module("src.pipelines.inference")
    pipeline = importlib.import_module("src.common.pipeline")

    assert callable(training.Training.run)
    assert callable(inference.Inference.run)
    assert hasattr(pipeline, "Pipeline")
