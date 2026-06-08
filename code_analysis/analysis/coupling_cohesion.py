"""
Simple coupling and cohesion analyzer for Python files in the
`esb_03_2026_tune_experiment` package.

Produces:
- coupling_cohesion_summary.csv : per-file cohesion and coupling counts
- coupling_diagram.mmd : Mermaid graph (module import edges)

Usage:
    python analysis/coupling_cohesion.py --folder . --outdir ./analysis/results

Cohesion heuristic (per-module):
- For each function, collect the set of attribute/global names it accesses.
- Cohesion score = (# of function pairs that share at least one name) / total function pairs
  (1.0 = highly cohesive, 0.0 = no shared attributes between functions)

Coupling heuristic (per-module):
- Count of unique imports that reference other modules in the same folder.
- Also outputs an import graph (edges module -> imported_module).

This is a static, best-effort analysis intended for developer guidance,
not a formal software metric tool.
"""

import ast
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Coupling and cohesion analyzer")
    p.add_argument("--folder", type=Path, default=Path('.'), help="Folder to analyze")
    p.add_argument("--outdir", type=Path, default=Path('./analysis/results'), help="Output directory")
    return p.parse_args()


class ModuleAnalysis:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.imports: Set[str] = set()
        # mapping func_name -> set of names accessed (attributes, globals)
        self.func_access: Dict[str, Set[str]] = {}
        # number of top-level functions
        self.func_count: int = 0


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.current_func: str = None
        self.func_access: Dict[str, Set[str]] = {}
        self.imports: Set[str] = set()

    def visit_Import(self, node):
        for n in node.names:
            name = (n.name or '').split('.')[0]
            if name:
                self.imports.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # from package.module import ...  -> package or module
        mod = node.module or ''
        if mod:
            name = mod.split('.')[0]
            self.imports.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_name = node.name
        prev = self.current_func
        self.current_func = func_name
        self.func_access.setdefault(func_name, set())
        # traverse body
        self.generic_visit(node)
        self.current_func = prev

    def visit_Attribute(self, node: ast.Attribute):
        # record attribute access like `x.y` -> 'y'
        try:
            attr = node.attr
            if self.current_func is not None and attr:
                self.func_access.setdefault(self.current_func, set()).add(attr)
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # record variable/global usage
        if self.current_func is not None and isinstance(node.ctx, (ast.Load, ast.Store)):
            self.func_access.setdefault(self.current_func, set()).add(node.id)
        self.generic_visit(node)


def analyze_file(path: Path) -> ModuleAnalysis:
    m = ModuleAnalysis(path)
    try:
        src = path.read_text(encoding='utf-8')
    except Exception:
        src = ''
    try:
        tree = ast.parse(src)
    except Exception:
        tree = None
    if tree is None:
        return m

    analyzer = Analyzer()
    analyzer.visit(tree)
    m.imports = analyzer.imports
    m.func_access = analyzer.func_access
    m.func_count = len([n for n in tree.body if isinstance(n, ast.FunctionDef)])
    return m


def cohesion_score(func_access: Dict[str, Set[str]]) -> float:
    funcs = list(func_access.keys())
    n = len(funcs)
    if n <= 1:
        return 1.0
    shared_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if func_access[funcs[i]] & func_access[funcs[j]]:
                shared_pairs += 1
    if total_pairs == 0:
        return 1.0
    return shared_pairs / total_pairs


def build_import_graph(analyses: List[ModuleAnalysis]) -> Dict[str, Set[str]]:
    names = {m.name for m in analyses}
    graph: Dict[str, Set[str]] = {m.name: set() for m in analyses}
    for m in analyses:
        for imp in m.imports:
            if imp in names and imp != m.name:
                graph[m.name].add(imp)
    return graph


def write_mermaid(graph: Dict[str, Set[str]], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["flowchart LR"]
    for src, targets in graph.items():
        if not targets:
            lines.append(f"    {src}[" + src + "]")
        for t in targets:
            lines.append(f"    {src} --> {t}")
    out.write_text("\n".join(lines), encoding='utf-8')


def write_summary(analyses: List[ModuleAnalysis], graph: Dict[str, Set[str]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'module', 'func_count', 'cohesion', 'coupling_count', 'imports'
        ])
        writer.writeheader()
        for m in analyses:
            coh = cohesion_score(m.func_access)
            coupling = len([imp for imp in m.imports if imp in graph and imp != m.name])
            writer.writerow({
                'module': m.name,
                'func_count': m.func_count,
                'cohesion': f"{coh:.3f}",
                'coupling_count': coupling,
                'imports': json.dumps(sorted(list(m.imports)))
            })


def main():
    args = parse_args()
    folder = args.folder.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # consider only top-level .py files in the provided folder
    py_files = sorted([p for p in folder.glob('*.py') if p.name != '__init__.py'])
    analyses: List[ModuleAnalysis] = []
    for p in py_files:
        analyses.append(analyze_file(p))

    graph = build_import_graph(analyses)

    write_summary(analyses, graph, outdir / 'coupling_cohesion_summary.csv')
    write_mermaid(graph, outdir / 'coupling_diagram.mmd')

    print(f"Wrote summary to {outdir / 'coupling_cohesion_summary.csv'}")
    print(f"Wrote mermaid diagram to {outdir / 'coupling_diagram.mmd'}")


if __name__ == '__main__':
    main()
