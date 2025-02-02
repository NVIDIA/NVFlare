from nvflare.edge.web.models.base_model import DictModel


class StudyResponse(DictModel):

    def __init__(self, status: str, session_id: str, study_id: str = None, job_id: str = None,
                 retry_wait: int = None, device_state: dict = None):
        self.status = status
        self.session_id = session_id
        self.study_id = study_id
        self.job_id = job_id
        self.retry_wait = retry_wait
        self.device_state = device_state

