"""
This module defines a machine learning pipeline for processing, training, and evaluating data.
The pipeline executes the following steps:
1. Prepare Sample Data: Preprocess raw data.
2. Feature Engineering: Compute features and store in the feature store.
3. Transform Sample Data: Perform transformations and retrieve features.
4. Train Model: Train a model using the features.
5. Predict: Use the trained model for predictions.
6. Score: Evaluate the model performance.
7. Register Model: Register the model and its metadata.
"""

from azure.identity import DefaultAzureCredential
import argparse
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import MLClient, Input, load_component
from azureml.featurestore import create_feature_set_spec
from azureml.featurestore.feature_source import CsvFeatureSource
from azureml.featurestore.contracts import (
    DateTimeOffset,
    TransformationCode,
    Column,
    ColumnType,
    TimestampColumn,
)
from azure.ai.ml.entities import (
    FeatureStore,
    FeatureSet,
    FeatureSetSpecification,
    DataColumn,
    DataColumnType,
    FeatureStoreEntity,
    MaterializationSettings,
    MaterializationComputeResource,
)
from mlops.common.get_compute import get_compute
from mlops.common.get_environment import get_environment
from mlops.common.config_utils import MLOpsConfig
from mlops.common.naming_utils import (
    generate_experiment_name,
    generate_model_name,
    generate_run_name,
)

gl_pipeline_components = []


def create_feature_store(config):
    """
    Create a FeatureStore object based on the configuration.

    Args:
        config: Configuration object containing feature store settings.

    Returns:
        FeatureStore: A configured FeatureStore object.
    """
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config.feature_store_config["subscription_id"],
        resource_group_name=config.feature_store_config["resource_group_name"],
    )

    fs = FeatureStore(
        name=config.feature_store_config["name"],
        location=config.feature_store_config["location"]
    )
    fs_poller = ml_client.feature_stores.begin_create(fs)
    return fs_poller.result()


def define_features(ml_client, feature_store, config):
    """
    Define and register feature sets for the feature store.

    Args:
        ml_client: The MLClient for communicating with Azure services.
        feature_store: The feature store where features will be stored.
        config: Configuration object with feature store data path and other settings.

    Returns:
        FeatureSetSpecification: The defined feature set specification.
    """
    feature_set_spec = create_feature_set_spec(
        source=CsvFeatureSource(
            path=config.feature_store_config["data_path"],
            timestamp_column=TimestampColumn(name="pickup_datetime"),
            source_delay=DateTimeOffset(days=0, hours=0, minutes=20),
        ),
        transformation_code=TransformationCode(
            path=config.feature_store_config["transformation_code_path"],
            transformer_class="TaxiDataTransformer",
        ),
        index_columns=[Column(name="vendorID", type=ColumnType.string)],
        source_lookback=DateTimeOffset(days=7, hours=0, minutes=0),
        temporal_join_lookback=DateTimeOffset(days=1, hours=0, minutes=0),
        infer_schema=True,
    )

    ml_client.feature_sets.create_or_update(feature_set_spec, feature_store_name=feature_store.name)
    return feature_set_spec


def register_entity(fs_client, entity_name, index_columns):
    """
    Register a feature store entity.

    Args:
        fs_client: The feature store client.
        entity_name: Name of the entity.
        index_columns: List of index columns for the entity.

    Returns:
        The result of the entity registration.
    """
    entity_config = FeatureStoreEntity(
        name=entity_name,
        version="1",
        index_columns=[DataColumn(name=col, type=DataColumnType.STRING) for col in index_columns],
        stage="Development",
        description=f"This entity represents {entity_name} index key(s).",
        tags={"data_type": "nonPII"},
    )

    poller = fs_client.feature_store_entities.begin_create_or_update(entity_config)
    return poller.result()


def register_feature_set(fs_client, feature_set_name, entity_name, specification_path):
    """
    Register a feature set with the feature store.

    Args:
        fs_client: The feature store client.
        feature_set_name: Name of the feature set.
        entity_name: Name of the associated entity.
        specification_path: Path to the feature set specification.

    Returns:
        The result of the feature set registration.
    """
    feature_set_config = FeatureSet(
        name=feature_set_name,
        version="1",
        description=f"Feature set for {feature_set_name}",
        entities=[f"azureml:{entity_name}:1"],
        stage="Development",
        specification=FeatureSetSpecification(path=specification_path),
        tags={"data_type": "nonPII"},
    )

    poller = fs_client.feature_sets.begin_create_or_update(feature_set_config)
    return poller.result()


def enable_materialization(fs_client, feature_set_name):
    """
    Enable materialization for a feature set.

    Args:
        fs_client: The feature store client.
        feature_set_name: Name of the feature set.

    Returns:
        The result of enabling materialization.
    """
    feature_set_config = fs_client.feature_sets.get(name=feature_set_name, version="1")

    feature_set_config.materialization_settings = MaterializationSettings(
        offline_enabled=True,
        resource=MaterializationComputeResource(instance_type="standard_e8s_v3"),
        spark_configuration={
            "spark.driver.cores": 4,
            "spark.driver.memory": "36g",
            "spark.executor.cores": 4,
            "spark.executor.memory": "36g",
            "spark.executor.instances": 2,
        },
        schedule=None,
    )

    poller = fs_client.feature_sets.begin_create_or_update(feature_set_config)
    return poller.result()


@pipeline()
def london_taxi_data_regression(pipeline_job_input, model_name, build_reference, feature_store_name):
    """
    Run a pipeline for regression analysis on London taxi data.

    Parameters:
    pipeline_job_input (str): Path to the input data.
    model_name (str): Name of the model.
    build_reference (str): Reference for the build.
    feature_store_name (str): Name of the feature store.

    Returns:
    dict: A dictionary containing paths to various data, the model, predictions, and score report.
    """
    prepare_sample_data = gl_pipeline_components[0](
        raw_data=pipeline_job_input,
    )
    feature_engineering = gl_pipeline_components[1](
        clean_data=prepare_sample_data.outputs.prep_data,
        feature_store_name=feature_store_name,
    )
    transform_sample_data = gl_pipeline_components[2](
        clean_data=prepare_sample_data.outputs.prep_data,
        feature_data=feature_engineering.outputs.feature_data,
    )
    train_with_sample_data = gl_pipeline_components[3](
        training_data=transform_sample_data.outputs.transformed_data,
        feature_store_name=feature_store_name,
    )
    predict_with_sample_data = gl_pipeline_components[4](
        model_input=train_with_sample_data.outputs.model_output,
        test_data=train_with_sample_data.outputs.test_data,
        feature_store_name=feature_store_name,
    )
    score_with_sample_data = gl_pipeline_components[5](
        predictions=predict_with_sample_data.outputs.predictions,
        model=train_with_sample_data.outputs.model_output,
    )
    gl_pipeline_components[6](
        model_metadata=train_with_sample_data.outputs.model_metadata,
        model_name=model_name,
        score_report=score_with_sample_data.outputs.score_report,
        build_reference=build_reference,
    )

    return {
        "pipeline_job_prepped_data": prepare_sample_data.outputs.prep_data,
        "pipeline_job_transformed_data": transform_sample_data.outputs.transformed_data,
        "pipeline_job_trained_model": train_with_sample_data.outputs.model_output,
        "pipeline_job_test_data": train_with_sample_data.outputs.test_data,
        "pipeline_job_predictions": predict_with_sample_data.outputs.predictions,
        "pipeline_job_score_report": score_with_sample_data.outputs.score_report,
    }


def construct_pipeline(
    cluster_name: str,
    environment_name: str,
    display_name: str,
    build_environment: str,
    build_reference: str,
    model_name: str,
    dataset_name: str,
    feature_store_name: str,
    ml_client,
):
    """
    Construct a pipeline job for London taxi data regression.

    Args:
        cluster_name (str): The name of the cluster to use for pipeline execution.
        environment_name (str): The name of the environment to use for pipeline execution.
        display_name (str): The display name of the pipeline job.
        build_environment (str): The environment to deploy the pipeline job.
        build_reference (str): The build reference for the pipeline job.
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.
        feature_store_name (str): The name of the feature store.
        ml_client: The machine learning client.

    Returns:
        pipeline_job: The constructed pipeline job.
    """
    registered_data_asset = ml_client.data.get(name=dataset_name, label="latest")

    parent_dir = os.path.join(os.getcwd(), "mlops/london_taxi/components")

    prepare_data = load_component(source=parent_dir + "/prep.yml")
    feature_engineering = load_component(source=parent_dir + "/feature_engineering.yml")
    transform_data = load_component(source=parent_dir + "/transform.yml")
    train_model = load_component(source=parent_dir + "/train.yml")
    predict_result = load_component(source=parent_dir + "/predict.yml")
    score_data = load_component(source=parent_dir + "/score.yml")
    register_model = load_component(source=parent_dir + "/register.yml")

    # Set the environment name to custom environment using name and version number
    prepare_data.environment = environment_name
    feature_engineering.environment = environment_name
    transform_data.environment = environment_name
    train_model.environment = environment_name
    predict_result.environment = environment_name
    score_data.environment = environment_name
    register_model.environment = environment_name

    gl_pipeline_components.extend([
        prepare_data,
        feature_engineering,
        transform_data,
        train_model,
        predict_result,
        score_data,
        register_model
    ])

    pipeline_job = london_taxi_data_regression(
        Input(type="uri_folder", path=registered_data_asset.id),
        model_name,
        build_reference,
        feature_store_name,
    )

    pipeline_job.display_name = display_name
    pipeline_job.tags = {
        "environment": build_environment,
        "build_reference": build_reference,
    }

    # demo how to change pipeline output settings
    pipeline_job.outputs.pipeline_job_prepped_data.mode = "rw_mount"

    # set pipeline level compute
    pipeline_job.settings.default_compute = cluster_name
    pipeline_job.settings.force_rerun = True
    # set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    return pipeline_job


def prepare_and_execute(
    build_environment: str,
    wait_for_completion: str,
    output_file: str,
):
    """
    Prepare and execute the MLOps pipeline.

    Args:
        build_environment (str): environment name to execute.
        wait_for_completion (str): "True" or "False" - indicates whether to wait for the job to complete.
        output_file (str): The path to the output file where the job name will be written.
    """
    model_name = "london_taxi"

    config = MLOpsConfig(environment=build_environment)

    ml_client = MLClient(
        DefaultAzureCredential(),
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
    )

    # Create or update feature store
    feature_store = create_feature_store(config)

    # Initialize feature store client
    fs_client = MLClient(
        DefaultAzureCredential(),
        config.feature_store_config["subscription_id"],
        config.feature_store_config["resource_group_name"],
        feature_store.name,
    )

    # Define and register features
    define_features(ml_client, feature_store, config)

    # Register entity
    register_entity(fs_client, "account", ["accountID"])

    # Register feature set
    register_feature_set(fs_client, "transactions", "account", config.feature_store_config["feature_set_spec_path"])

    # Enable materialization
    enable_materialization(fs_client, "transactions")

    pipeline_config = config.get_pipeline_config(model_name)

    compute = get_compute(
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
        pipeline_config["cluster_name"],
        pipeline_config["cluster_size"],
        pipeline_config["cluster_region"],
    )

    environment = get_environment(
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
        config.environment_configuration["env_base_image"],
        pipeline_config["conda_path"],
        pipeline_config["aml_env_name"],
    )

    print(f"Environment: {environment.name}, version: {environment.version}")

    published_model_name = generate_model_name(model_name)
    published_experiment_name = generate_experiment_name(model_name)
    published_run_name = generate_run_name(
        config.environment_configuration["build_reference"]
    )

    pipeline_job = construct_pipeline(
        compute.name,
        f"azureml:{environment.name}:{environment.version}",
        published_run_name,
        build_environment,
        config.environment_configuration["build_reference"],
        published_model_name,
        pipeline_config["dataset_name"],
        feature_store.name,
        ml_client,
    )

    execute_pipeline(
        config.aml_config["subscription_id"],
        config.aml_config["resource_group_name"],
        config.aml_config["workspace_name"],
        published_experiment_name,
        pipeline_job,
        wait_for_completion,
        output_file,
    )


def execute_pipeline(
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
    experiment_name: str,
    pipeline_job: pipeline,
    wait_for_completion: str,
    output_file: str,
):
    """
    Execute a pipeline job in Azure Machine Learning service.

    Args:
        subscription_id (str): The Azure subscription ID.
        resource_group_name (str): The name of the resource group.
        workspace_name (str): The name of the Azure Machine Learning workspace.
        experiment_name (str): The name of the experiment.
        pipeline_job (pipeline): The pipeline job to be executed.
        wait_for_completion (str): "True" or "False" - indicates whether to wait for the job to complete.
        output_file (str): The path to the output file where the job name will be written.

    Raises:
        Exception: If the job fails to complete.

    Returns:
        None
    """
    try:
        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        pipeline_job = client.jobs.create_or_update(
            pipeline_job, experiment_name=experiment_name
        )

        print(f"The job {pipeline_job.name} has been submitted!")
        if output_file is not None:
            with open(output_file, "w") as out_file:
                out_file.write(pipeline_job.name)

        if wait_for_completion == "True":
            client.jobs.stream(pipeline_job.name)

    except Exception as ex:
        print(
            "Oops! invalid credentials or error while creating ML environment.. Try again...",
            ex,
        )
        raise


def main():
    """Parse the command line arguments and call the `prepare_and_execute` function."""
    parser = argparse.ArgumentParser("build_environment")
    parser.add_argument(
        "--build_environment",
        type=str,
        help="configuration environment for the pipeline",
    )
    parser.add_argument(
        "--wait_for_completion",
        type=str,
        help="determine if pipeline to wait for job completion",
        default="True",
    )
    parser.add_argument(
        "--output_file", type=str, required=False, help="A file to save run id"
    )
    args = parser.parse_args()

    prepare_and_execute(
        args.build_environment, args.wait_for_completion, args.output_file
    )


if __name__ == "__main__":
    main()
