from nvflare.app_common.abstract.generic_task import GenericTask
from nvflare.app_common.workflows.model_controller import ModelController


class GenericTaskController(ModelController):
    def __init__(self, task_name: str = "gen_task", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name

    def run(self):
        self.info(f"{self.task_name} task started.")

        # use FLModel structure, add empty model
        task = GenericTask()
        clients = self.sample_clients()
        results = self.send_task_and_wait(task_name=self.task_name, targets=clients, data=task)

        print(f"\n\n =================== controller result = {results} ")

        self.info("Finished etl.")

    def send_task_and_wait(self, task_name, targets, data):
        return self.send_model_and_wait(task_name=task_name, targets=targets, data=data)
