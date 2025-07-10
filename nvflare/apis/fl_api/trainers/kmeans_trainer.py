class KMeansTrainer(FedTrainer):
    """FedTrainer for scikit-learn KMeans models."""

    def __init__(self, model: BaseEstimator, **kwargs):
        self.model = model

        super().__init__(
            local_trainer=self.model, get_state_fn=self._get_model_state, set_state_fn=self._set_model_state, **kwargs
        )

    def _get_model_state(self) -> Dict[str, Any]:
        return {
            "cluster_centers_": self.model.cluster_centers_.tolist(),
            "n_features_in_": self.model.n_features_in_,  # type: ignore
            "n_clusters": self.model.n_clusters,
            "n_iter_": self.model.n_iter_,
            "inertia_": self.model.inertia_,
            "_n_threads": self.model._n_threads,  # type: ignore
        }

    def _set_model_state(self, state: Dict[str, Any]) -> None:
        import numpy as np

        self.model.cluster_centers_ = np.array(state["cluster_centers_"])
        self.model.n_features_in_ = state["n_features_in_"]  # type: ignore
        self.model.n_clusters = state["n_clusters"]
        self.model.n_iter_ = state["n_iter_"]
        self.model.inertia_ = state["inertia_"]
        self.model._n_threads = state["_n_threads"]  # type: ignore
