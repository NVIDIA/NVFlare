class XGBoostTrainer(FedTrainer):
    """FedTrainer for XGBoost models."""

    def __init__(self, model: XGBModel, **kwargs):
        self.model = model

        super().__init__(
            local_trainer=self.model, get_state_fn=self._get_model_state, set_state_fn=self._set_model_state, **kwargs
        )

    def _get_model_state(self) -> Dict[str, Any]:
        return self.model.get_booster().save_raw("json").decode()

    def _set_model_state(self, state: str) -> None:
        self.model.load_model(bytearray(state, "utf-8"))
