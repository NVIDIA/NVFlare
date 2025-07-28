import json
import os

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.edge_job import EdgeJob
from nvflare.edge.models.model import DeviceModel
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.widgets.evaluator import GlobalEvaluator
from nvflare.job_config.file_source import FileSource

_TRAINER_NAME = "trainer"
_DEVICE_CONFIG_FILE_NAME = "device_config.json"


class EdgeJobMaker:

    def __init__(
        self,
        job_name: str,
        device_model: DeviceModel,
        max_num_active_model_versions: int = 3,
        max_model_version: int = 20,
        update_timeout: int = 5.0,
        num_updates_for_model: int = 100,
        max_model_history: int = 10,
        global_lr: float = 0.0001,
        staleness_weight: bool = False,
        device_selection_size: int = 100,
        min_hole_to_fill: int = 1,
        device_reuse: bool = True,
        const_selection: bool = False,
        custom_source_root: str = None,
    ):
        if not isinstance(device_model, DeviceModel):
            raise ValueError(f"model must be a DeviceModel but got {type(device_model)}")

        if custom_source_root and not os.path.isdir(custom_source_root):
            raise ValueError(f"{custom_source_root} is not a valid directory")

        self.job_name = job_name
        self.method_name = "edge"
        self.device_model = device_model
        self.pt_model = device_model.net
        self.max_num_active_model_versions = max_num_active_model_versions
        self.max_model_version = max_model_version
        self.update_timeout = update_timeout
        self.num_updates_for_model = num_updates_for_model
        self.max_model_history = max_model_history
        self.global_lr = global_lr
        self.staleness_weight = staleness_weight
        self.device_selection_size = device_selection_size
        self.min_hole_to_fill = min_hole_to_fill
        self.device_reuse = device_reuse
        self.const_selection = const_selection
        self.device_trainer_args = None
        self.custom_source_root = custom_source_root
        self.job = EdgeJob(name=self.job_name, edge_method=self.method_name)

    def set_device_trainer(self, **kwargs):
        self.device_trainer_args = kwargs

    def set_evaluator(
        self,
        torchvision_dataset,
        eval_frequency: int = 1,
    ):
        evaluator = GlobalEvaluator(
            model_path=self.pt_model,
            torchvision_dataset=torchvision_dataset,
            eval_frequency=eval_frequency,
        )
        self.job.to_server(evaluator, id="evaluator")

    def configure_simulation(
        self,
        task_processor: DeviceTaskProcessor,
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        self.job.configure_simulation(task_processor, job_timeout, num_devices, num_workers)

    def make(self, result_dir):
        # use EdgeJob to create a job
        job = self.job

        factory = ModelUpdateDXOAggrFactory()
        job.configure_client(
            aggregator_factory=factory,
            max_model_versions=self.max_num_active_model_versions,
            update_timeout=self.update_timeout,
        )

        # add persistor, model_manager, and device_manager
        persistor_id = job.to_server(PTFileModelPersistor(model=self.pt_model), id="persistor")

        model_manager = BuffModelManager(
            num_updates_for_model=self.num_updates_for_model,
            max_model_history=self.max_model_history,
            global_lr=self.global_lr,
            staleness_weight=self.staleness_weight,
        )
        model_manager_id = job.to_server(model_manager, id="model_manager")

        device_manager = BuffDeviceManager(
            device_selection_size=self.device_selection_size,
            min_hole_to_fill=self.min_hole_to_fill,
            device_reuse=self.device_reuse,
            const_selection=self.const_selection,
        )
        device_manager_id = job.to_server(device_manager, id="device_manager")

        # add model_update_assessor
        assessor = ModelUpdateAssessor(
            persistor_id=persistor_id,
            model_manager_id=model_manager_id,
            device_manager_id=device_manager_id,
            max_model_version=self.max_model_version,
        )
        job.configure_server(
            assessor=assessor,
        )

        # create device config file
        if not self.device_trainer_args:
            raise RuntimeError("device trainer must be defined")

        trainer_config = {"type": "Trainer.DLTrainer", "name": _TRAINER_NAME, "args": self.device_trainer_args}

        device_config = {"components": [trainer_config], "executors": {"train": f"@{_TRAINER_NAME}"}}

        with open(_DEVICE_CONFIG_FILE_NAME, "w") as f:
            json.dump(device_config, f, indent=4)

        job.to_server(FileSource(_DEVICE_CONFIG_FILE_NAME, app_folder_type="config"))

        if self.custom_source_root:
            job.to_server(self.custom_source_root)
            job.to_clients(self.custom_source_root)

        job.export_job(result_dir)
