def _get_model_weights(self) -> Shareable:
    # Get state dict and send as weights
    new_weights = self.model.state_dict()
    new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

    outgoing_dxo = DXO(
        data_kind=DataKind.WEIGHTS, data=new_weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
    )
    return outgoing_dxo.to_shareable()
