"""Execute the solution cells with real images and explicitly bounded training."""
import argparse
import ast
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", type=Path, help="Extracted ebay_boys_girls_shirts directory")
parser.add_argument("--full", action="store_true", help="Use the notebook's original sample sizes and epochs")
parser.add_argument("--samples-per-class", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()
if args.samples_per_class < 2 or args.epochs < 1:
    parser.error("Use at least two samples per class and one epoch")
dataset = args.dataset.expanduser().resolve()
for name in ("boys_train.csv", "boys_test.csv", "girls_train.csv", "girls_test.csv"):
    if not (dataset / name).is_file():
        parser.error("Missing dataset file: " + name)
os.environ["SHIRTS_DATASET"] = str(dataset)

class BoundedTraining(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "get_final_matrices":
            node.keywords = [ast.keyword(arg="n_train", value=ast.Constant(args.samples_per_class)),
                             ast.keyword(arg="n_test", value=ast.Constant(min(16, args.samples_per_class)))]
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            for keyword in node.keywords:
                if keyword.arg == "epochs": keyword.value = ast.Constant(args.epochs)
                if keyword.arg == "batch_size": keyword.value = ast.Constant(8)
                if keyword.arg == "verbose": keyword.value = ast.Constant(0)
        return node

root = Path(__file__).resolve().parents[1]
notebook = next((root / "solutions").glob("*.ipynb"))
cells = json.loads(notebook.read_text())["cells"]
namespace = {"__name__": "__main__"}
completed = 0
with tempfile.TemporaryDirectory(prefix="image-classification-") as directory:
    old_cwd = Path.cwd()
    try:
        os.chdir(directory)
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code": continue
            source = "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))
            if not source.strip(): continue
            tree = ast.parse(source)
            if not args.full: tree = BoundedTraining().visit(tree)
            ast.fix_missing_locations(tree)
            print("Executing cell %d" % index, flush=True)
            exec(compile(tree, "%s:cell%d" % (notebook.name, index), "exec"), namespace)
            completed += 1
            if "plt" in namespace: namespace["plt"].close("all")
        import numpy as np
        assert np.isfinite(namespace["x_train"]).all()
        model = namespace["model"]
        predictions = model.predict(namespace["x_test"], verbose=0)
        assert predictions.shape[0] == len(namespace["y_test"])
        assert np.isfinite(predictions).all()
        assert ((predictions >= 0) & (predictions <= 1)).all()
    finally:
        os.chdir(old_cwd)
print("PASS: %d code cells; %s" % (completed, "original training sizes" if args.full else "bounded real-image training, not historical-score reproduction"))
