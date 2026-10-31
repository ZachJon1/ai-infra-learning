from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

base = Path("model_repository/toy_classifier")

for version in (1, 2):
    weights = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32) * version

    node = helper.make_node(
        "MatMul",
        inputs=["INPUT", "W"],
        outputs=["OUTPUT"]
    )

    graph = helper.make_graph(
        [node],
        f"toy_classifier_v{version}",
        [helper.make_tensor_value_info(
            "INPUT", TensorProto.FLOAT, ["batch", 4]
        )],
        [helper.make_tensor_value_info(
            "OUTPUT", TensorProto.FLOAT, ["batch", 2]
        )],
        [numpy_helper.from_array(weights, name="W")]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=8
    )

    onnx.checker.check_model(model)

    destination = base / str(version) / "model.onnx"
    onnx.save(model, str(destination))
    print(f"Created {destination}")