"""Promote mixed int32 shape inputs to int64 for TensorRT 10.

ONNX Shape produces int64, and TensorRT 10's shape math also uses int64.
This export still feeds some int32 constants into Concat and Mul. Casting
those int32 inputs to int64 makes the node legal without changing values.
Gather and Reshape are left alone: their inputs are allowed to differ.
"""

import sys

import onnx
from onnx import TensorProto, helper


def collect_types(model):
    types = {}
    for initializer in model.graph.initializer:
        types[initializer.name] = initializer.data_type
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        elem_type = value_info.type.tensor_type.elem_type
        if elem_type:
            types[value_info.name] = elem_type
    return types


def cast_to_int64(pending_nodes, created, tensor_name):
    output_name = tensor_name + "_to_int64"
    if output_name not in created:
        pending_nodes.append(
            helper.make_node(
                "Cast",
                [tensor_name],
                [output_name],
                to=TensorProto.INT64,
                name="Cast_" + output_name,
            )
        )
        created.add(output_name)
    return output_name


def fix_model(src_path, dst_path):
    model = onnx.shape_inference.infer_shapes(onnx.load(src_path))
    types = collect_types(model)
    created = set()
    rewritten = []
    fixed = 0
    for node in model.graph.node:
        if node.op_type in ("Concat", "Mul"):
            input_types = [types.get(name) for name in node.input]
            integer_types = {dtype for dtype in input_types if dtype in (TensorProto.INT32, TensorProto.INT64)}
            if len(integer_types) > 1:
                casts = []
                new_inputs = []
                for name, dtype in zip(node.input, input_types):
                    if dtype == TensorProto.INT32:
                        new_inputs.append(cast_to_int64(casts, created, name))
                    else:
                        new_inputs.append(name)
                rewritten.extend(casts)
                del node.input[:]
                node.input.extend(new_inputs)
                fixed += 1
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    types = collect_types(model)
    mixed = []
    for node in model.graph.node:
        if node.op_type not in ("Concat", "Mul"):
            continue
        integer_types = {
            types.get(name)
            for name in node.input
            if types.get(name) in (TensorProto.INT32, TensorProto.INT64)
        }
        if len(integer_types) > 1:
            mixed.append(node.name)
    if mixed:
        raise RuntimeError(f"still mixed integer nodes: {mixed[:10]}")

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    print(f"{src_path} -> {dst_path}: fixed {fixed} nodes, added {len(created)} casts")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} src.onnx dst.onnx")
    fix_model(sys.argv[1], sys.argv[2])
