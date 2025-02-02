from nvflare.edge.web.models.base_model import DictModel


class TaskRequest(DictModel):
    def __init__(self, session_id: str, study_id: str, device_state: dict = None):
        self.session_id = session_id
        self.study_id = study_id
        self.device_state = device_state

