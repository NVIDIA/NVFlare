from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward


def export_model(net, input_tensor_example, label_tensor_example):
    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = export(net, (input_tensor_example, label_tensor_example), strict=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch()
    return ep
