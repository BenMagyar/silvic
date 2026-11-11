import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from neuronxcc import nki

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


compiler = module("silvic_neuron_compiler", ROOT / "compiler" / "compile.py")
serving = module("silvic_neuron_serving", ROOT / "kernels" / "serving.py")
training = module("silvic_neuron_training", ROOT / "kernels" / "training.py")


class KernelContracts(unittest.TestCase):
    def test_signed_leaf_sums_preserve_cancellation_across_i64_limits(self):
        rows = 128
        partition = np.arange(rows, dtype=np.int32).reshape(rows, 1)
        values = [((1 << 63) - 1, -(1 << 63)) * (rows // 2), (1, -1) * (rows // 2)]
        limbs = np.empty((8, rows, 1), dtype=np.int32)
        for component in range(2):
            for row, value in enumerate(values[component]):
                for digit in range(4):
                    limbs[component * 4 + digit, row, 0] = (
                        value >> 48 if digit == 3 else (value >> (16 * digit)) & 65535
                    )
        meta = np.zeros((8, 1), dtype=np.int32)
        meta[1] = rows
        partials = nki.jit(
            training.leaf_sums.func, mode="simulation", kernel_return=True
        )(partition, limbs, meta, ROWS=rows, TILE=128, CHUNKS=2, LIMBS=8)
        for component in range(2):
            reconstructed = sum(
                int(partials[:, component * 4 + digit, 0].sum(dtype=np.float64))
                << (16 * digit)
                for digit in range(4)
            )
            self.assertEqual(reconstructed, sum(values[component]))

    def test_serving_tree_order_and_single_core_ports_on_each_target(self):
        x = np.zeros((1, 32), dtype=np.float32)
        metadata = np.array(
            [[0, -1, -1, 0, 0, 0, 1, tree, 1] for tree in range(3)], dtype=np.int32
        ).reshape(1, -1)
        values = np.array(
            [[0, 16777216, 16777216, 0, 1, 1, 0, -16777216, -16777216]],
            dtype=np.float32,
        )
        categories = np.zeros((1, 1), dtype=np.uint32)
        prefix = np.array([[3]], dtype=np.int32)
        args = (x, metadata, values, categories, prefix)
        constants = {"RECORDS": 3, "ROWS": 32}
        simulate = nki.jit(serving.score.func, mode="simulation", kernel_return=True)
        for count, expected in enumerate((0, 16777216, 16777216, 0)):
            prefix[0, 0] = count
            np.testing.assert_array_equal(
                simulate(*args, **constants), np.full_like(x, expected)
            )
        inputs = [
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in zip(
                ("input", "metadata", "values", "categories", "prefix"), args
            )
        ]
        outputs = [{"name": "scores", "dtype": "float32", "shape": [1, 32]}]
        for target in ("trn1", "inf2", "trn2"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                manifest = compiler.compile_kernel(
                    ROOT / "kernels" / "serving.py",
                    {
                        "entry": "score",
                        "target": target,
                        "constants": constants,
                        "inputs": inputs,
                        "outputs": outputs,
                    },
                    directory,
                )
                self.assertEqual(manifest["target"], target)
                self.assertEqual(
                    [
                        {key: port[key] for key in ("name", "dtype", "shape")}
                        for port in manifest["inputs"]
                    ],
                    inputs,
                )
                self.assertEqual(
                    [
                        {key: port[key] for key in ("name", "dtype", "shape")}
                        for port in manifest["outputs"]
                    ],
                    outputs,
                )


if __name__ == "__main__":
    unittest.main()
