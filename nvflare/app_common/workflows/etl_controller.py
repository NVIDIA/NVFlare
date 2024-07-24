from nvflare.app_common.workflows.generic_task_controller import GenericTaskController


class ETLController(GenericTaskController):
    def __init__(self, task_name="etl", *args, **kwargs):
        kwargs["task_name"] = task_name
        super().__init__(*args, **kwargs)
