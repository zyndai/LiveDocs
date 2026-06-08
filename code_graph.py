"""Code knowledge graph: built once at index time, loaded into RAM at server
start, queried with O(1) adjacency lookups per request.

Nodes  = symbols (functions/methods/classes/types), keyed by a qualified id
         "repo:file::symbol".
Edges  = CALLS (symbol -> symbol, resolved by name) and IMPORTS (file -> module).

Call edges are resolved by *name* (cheap, language-agnostic). Full type-aware
resolution isn't worth the cost here -- name resolution plus the vector layer
covers the question types this system serves. Ambiguous names (same name in
several files) link to all candidates; the reranker sorts it out downstream.

Runtime never traverses to compute anything heavy -- it reads precomputed
adjacency dicts. expand() returns 1-hop (configurable) neighbor chunk ids for
context expansion.
"""
import pickle

import networkx as nx

from config import GRAPH_PATH, GRAPH_EXPAND_HOPS, GRAPH_MAX_NEIGHBORS


def qualified_id(repo, file, symbol):
    return f"{repo}:{file}::{symbol}"


def build_graph(chunks, imports_by_file):
    """Build a DiGraph from chunk dicts (code_chunker output).

    chunks: list of chunk dicts with repo/file/symbol/calls/...
    imports_by_file: {(repo, file): [module, ...]}
    Returns a networkx.DiGraph with node attrs and CALLS/IMPORTS edges.
    """
    g = nx.DiGraph()

    # name -> [node_id, ...] so call edges can resolve a bare callee name.
    name_index = {}

    # First pass: create one node per distinct symbol (skip import pseudo-chunks).
    for c in chunks:
        if c["symbol_type"] == "imports":
            continue
        nid = qualified_id(c["repo"], c["file"], c["symbol"])
        if nid not in g:
            g.add_node(
                nid,
                repo=c["repo"],
                file=c["file"],
                symbol=c["symbol"],
                symbol_type=c["symbol_type"],
                lang=c["lang"],
                start_line=c["start_line"],
                end_line=c["end_line"],
            )
            name_index.setdefault(c["symbol"], []).append(nid)

    # Second pass: CALLS edges, resolved by callee name.
    for c in chunks:
        if c["symbol_type"] == "imports":
            continue
        src_id = qualified_id(c["repo"], c["file"], c["symbol"])
        for callee in c.get("calls", []):
            for tgt_id in name_index.get(callee, []):
                if tgt_id != src_id:
                    g.add_edge(src_id, tgt_id, kind="calls")

    # IMPORTS edges: file node -> module string (module nodes are lightweight).
    for (repo, file), modules in imports_by_file.items():
        file_node = f"{repo}:{file}"
        if file_node not in g:
            g.add_node(file_node, symbol_type="file", repo=repo, file=file)
        for mod in modules:
            mod_node = f"module::{mod}"
            if mod_node not in g:
                g.add_node(mod_node, symbol_type="module", symbol=mod)
            g.add_edge(file_node, mod_node, kind="imports")

    return g


def save_graph(g, path=None):
    path = path or GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)
    return path


class CodeGraph:
    """Runtime wrapper. Load once at server start; all methods are cheap reads."""

    def __init__(self, graph):
        self.g = graph
        # name -> [node_id] index for symbol-name lookups (rebuilt on load).
        self._by_name = {}
        for nid, data in graph.nodes(data=True):
            sym = data.get("symbol")
            if sym:
                self._by_name.setdefault(sym, []).append(nid)

    @classmethod
    def load(cls, path=None):
        path = path or GRAPH_PATH
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def ids_for_name(self, name):
        return self._by_name.get(name, [])

    def neighbors(self, node_id, hops=None, max_neighbors=None):
        """Return neighbor node ids within `hops` (callers + callees), bounded.

        O(neighbors) read -- no global traversal. Used for context expansion.
        """
        hops = GRAPH_EXPAND_HOPS if hops is None else hops
        max_neighbors = GRAPH_MAX_NEIGHBORS if max_neighbors is None else max_neighbors
        if node_id not in self.g:
            return []
        seen = set()
        frontier = {node_id}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                # callees (successors) + callers (predecessors)
                for m in list(self.g.successors(n)) + list(self.g.predecessors(n)):
                    if m not in seen and m != node_id:
                        seen.add(m)
                        nxt.add(m)
            frontier = nxt
            if not frontier:
                break
        # Drop module pseudo-nodes from expansion (not real chunks).
        out = [n for n in seen if not n.startswith("module::") and "::" in n]
        return out[:max_neighbors]

    def node(self, node_id):
        return self.g.nodes.get(node_id)

    def stats(self):
        calls = sum(1 for _, _, d in self.g.edges(data=True) if d.get("kind") == "calls")
        imports = sum(1 for _, _, d in self.g.edges(data=True) if d.get("kind") == "imports")
        return {"nodes": self.g.number_of_nodes(), "calls_edges": calls, "imports_edges": imports}
