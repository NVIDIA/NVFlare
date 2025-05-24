from nvflare.edge.aggregators.num_dxo_factory import NumDXOAggrFactory
from nvflare.edge.assessors.async_num import AsyncNumAssessor
from nvflare.edge.edge_job import EdgeJob

job = EdgeJob(
    name="num_async_job",
    edge_method="cnn",
)

factory = NumDXOAggrFactory()
job.configure_client(
    aggregator_factory=factory, max_model_versions=3, simulation_config_file="configs/num_async_config.json"
)

job.configure_server(
    assessor=AsyncNumAssessor(
        num_updates_for_model=10,
        max_model_history=3,
        max_model_version=10,
        device_selection_size=30,
    )
)

job.export_job("/tmp/nvflare/jobs/")
