"""Verify the supplied source files against their author-provided hash receipts."""
import argparse
import ast
from pathlib import Path

from dataset_io import load_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def triplet_ast(path, paper_endpoint=False):
    source = Path(path).read_text(encoding="utf-8")
    if paper_endpoint:
        source = source.replace("if bounds[0]<=ratio<bounds[1] and target_area*(1-ratio)>=64:",
                                "if bounds[0] <= ratio and (ratio <= bounds[1] if lev == 'H' else ratio < bounds[1]) and target_area*(1-ratio)>=64:")
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name == "make_triplet"]
    if len(functions) != 1:
        raise ValueError(f"Expected one make_triplet in {path}")
    return ast.dump(functions[0], include_attributes=False)


def verify_bundle(root=ROOT):
    code = Path(root) / "generation_code"
    receipt = load_json(code / "CODE_PROVENANCE.json")
    build = load_json(code / "ORIGINAL_BUILD_COMPLETE.json")
    checks = {}
    for name, expected in receipt["verified_original_runtime_sha256"].items():
        actual = sha256(code / "original_runtime" / name)
        if actual != expected or actual != build["code_sha256"][name]:
            raise ValueError(f"Historical source hash mismatch: {name}")
        checks["original_runtime/" + name] = actual
    legacy = sha256(code / "legacy/tools/build_protocol.py")
    if legacy != receipt["legacy_sha256"]:
        raise ValueError("Legacy helper differs from the supplied provenance receipt")
    checks["legacy/tools/build_protocol.py"] = legacy
    if triplet_ast(code / "build_occlusion_only.py") != triplet_ast(code / "original_runtime/build_lite.py", paper_endpoint=True):
        raise ValueError("The adapter differs beyond the documented paper-standard H endpoint")
    return {"status": "passed", "sha256": checks, "triplet_ast_equal_after_documented_H_endpoint_change": True,
            "verification_basis": "author-provided receipts; not an independent historical execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    import json
    print(json.dumps(verify_bundle(args.root), indent=2))


if __name__ == "__main__":
    main()
