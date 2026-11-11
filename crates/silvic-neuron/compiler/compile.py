import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

import neuronxcc

COMPILER_VERSION = "2.21.18209.0+043b1bf7"


def compile_kernel(source_path, request, output_directory):
    import numpy as np
    from neuronxcc.nki.compiler.backends.neuron.NumpyKernel import BaremetalKernel

    class CompileOnlyKernel(BaremetalKernel):
        def execute_neff(self, neff, ir, boundargs):
            self.compiled_neff = Path(neff)

    if neuronxcc.__version__ != COMPILER_VERSION:
        raise RuntimeError(
            f"Expected neuronx-cc {COMPILER_VERSION}, found {neuronxcc.__version__}"
        )
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="nki-", dir=output_directory))
    spec = importlib.util.spec_from_file_location("_silvic_nki_kernel", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, request["entry"])
    function = getattr(function, "func", function)
    neff = work / "model.neff"
    artifacts = work / "artifacts"
    options = "--auto-cast none"
    if request["target"] == "trn2":
        options += " --logical-nc-config 1"
    kernel = CompileOnlyKernel(
        func=function,
        func_name=request["entry"],
        kernel_return=True,
        platform_target=request["target"],
        additional_compile_opt=options,
        save_neff_name=str(neff),
        artifacts_dir=str(artifacts),
    )
    examples = [
        np.broadcast_to(np.zeros((), dtype=np.dtype(tensor["dtype"])), tensor["shape"])
        for tensor in request["inputs"]
    ]
    kernel(*examples, **request.get("constants", {}))
    if not neff.is_file() or neff.stat().st_size == 0:
        raise RuntimeError("Neuron compiler did not produce a NEFF")
    graph = json.loads((artifacts / "neff.json").read_text())
    shapes = graph["attrs"]["shape"][1]
    dtypes = graph["attrs"]["dltype"][1]

    def port(node_index, output_index):
        flat_index = graph["node_row_ptr"][node_index] + output_index
        return {
            "port": graph["nodes"][node_index]["output_names"][output_index],
            "dtype": dtypes[flat_index],
            "shape": shapes[flat_index],
        }

    def mapped_ports(logical, actual):
        if len(logical) != len(actual):
            raise RuntimeError(
                f"Port count mismatch: declared {len(logical)}, compiled {len(actual)}"
            )
        result = []
        for wanted, compiled in zip(logical, actual):
            if (
                wanted["shape"] != compiled["shape"]
                or wanted["dtype"] != compiled["dtype"]
            ):
                raise RuntimeError(
                    f"Port specialization mismatch: {wanted!r} versus {compiled!r}"
                )
            result.append({"name": wanted["name"], **compiled})
        return result

    manifest = {
        "compiler_version": neuronxcc.__version__,
        "target": request["target"],
        "neff": str(neff),
        "inputs": mapped_ports(
            request["inputs"], [port(index, 0) for index in graph["arg_nodes"]]
        ),
        "outputs": mapped_ports(
            request["outputs"], [port(head[0], head[1]) for head in graph["heads"]]
        ),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--request")
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    if args.version:
        print(neuronxcc.__version__)
        return
    if not all((args.source, args.request, args.output_directory)):
        parser.error("--source, --request and --output-directory are required")
    request = json.loads(Path(args.request).read_text())
    manifest = compile_kernel(args.source, request, args.output_directory)
    destination = Path(args.output_directory) / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
