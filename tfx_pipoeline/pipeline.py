import os
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration import pipeline
from tfx.proto import trainer_pb2
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.proto import pusher_pb2
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common.importer import Importer
from tfx.types import standard_artifacts

from tfx.components.trainer.executor import GenericExecutor

_pipeline_name = "pdm_pipeline"
_pipeline_root = os.path.join(os.getcwd(), "tfx_pipelines", _pipeline_name)
_data_root = os.path.join(os.getcwd(), "data")  # CSV input folder
_module_file = os.path.join(os.getcwd(), "tfx_pipeline", "modules.py")
_serving_model_dir = os.path.join(os.getcwd(), "serving_model", _pipeline_name)

def create_pipeline():
    # 1. ExampleGen (from CSV)
    example_gen = CsvExampleGen(input_base=_data_root)

    # 2. Stats + Schema + Validator
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )

    # 3. Transform (feature engineering logic)
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=_module_file
    )

    # 4. Trainer (with XGBoost)
    trainer = Trainer(
        module_file=_module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
    )

    # 5. Evaluator
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
    )

    # 6. Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen,
            example_validator, transform, trainer,
            evaluator, pusher
        ],
        enable_cache=True,
        metadata_connection_config=None
    )

if __name__ == "__main__":
    LocalDagRunner().run(create_pipeline())
