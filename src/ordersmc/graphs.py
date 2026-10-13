"""Graph summaries used by the paper experiments.

The original :func:`dag_from_edge_probs` helper is intentionally retained as a
fast, backwards-compatible heuristic.  Confirmatory experiments should use
:func:`optimal_bayes_action_dag`, which solves the declared constrained
decision problem globally, and :func:`dag_to_cpdag` for Markov-equivalence
summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import warnings
from typing import Dict, FrozenSet, Iterable, List, Literal, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize._highspy._core import (  # type: ignore[import-untyped]
    HIGHS_VERSION_MAJOR,
    HIGHS_VERSION_MINOR,
    HIGHS_VERSION_PATCH,
)
from scipy.sparse import coo_matrix, vstack

Edge = Tuple[str, str]

_EXACT_SUBSET_DP_MAX_NODES = 10
# Preserve the objective exactly in binary64, while making small nonzero
# utility differences visible to HiGHS' absolute cost tolerances. This is a
# numerical unit conversion, not an edge-dependent tie-breaking perturbation.
_MILP_OBJECTIVE_SCALE_EXPONENT = 20
_HIGHS_VERSION = (
    f"{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}"
)


@dataclass(frozen=True)
class CPDAG:
    """A completed partially directed acyclic graph.

    ``directed_edges`` contains compelled orientations. ``undirected_edges``
    contains reversible adjacencies, represented as two-element frozensets.
    """

    nodes: Tuple[str, ...]
    directed_edges: FrozenSet[Edge]
    undirected_edges: FrozenSet[FrozenSet[str]]

    def skeleton_edges(self) -> FrozenSet[FrozenSet[str]]:
        directed = {frozenset((parent, child)) for parent, child in self.directed_edges}
        return frozenset(directed | set(self.undirected_edges))

    def is_directed(self, parent: str, child: str) -> bool:
        return (parent, child) in self.directed_edges

    def is_undirected(self, left: str, right: str) -> bool:
        return frozenset((left, right)) in self.undirected_edges

    def to_networkx(self) -> nx.MultiDiGraph:
        """Return a lossless NetworkX representation.

        Directed CPDAG edges appear once with ``kind='directed'``. Reversible
        adjacencies appear in both directions with ``kind='undirected'``.
        """

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self.nodes)
        for parent, child in sorted(self.directed_edges):
            graph.add_edge(parent, child, kind="directed")
        for pair in sorted((tuple(sorted(pair)) for pair in self.undirected_edges)):
            left, right = pair
            graph.add_edge(left, right, kind="undirected")
            graph.add_edge(right, left, kind="undirected")
        return graph


def dag_from_edge_probs(
    edge_probs: Dict[Edge, float],
    nodes: Sequence[str],
    *,
    threshold: float = 0.5,
) -> nx.DiGraph:
    """Greedily retain high-probability edges while preserving acyclicity."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (parent, child), probability in sorted(edge_probs.items(), key=lambda item: -item[1]):
        if probability < threshold:
            break
        if parent == child or graph.has_edge(parent, child):
            continue
        if nx.has_path(graph, child, parent):
            continue
        graph.add_edge(parent, child, posterior_probability=float(probability))
    return graph


def edges_above_threshold(
    edge_probs: Dict[Edge, float],
    nodes: Sequence[str],
    *,
    threshold: float = 0.5,
) -> List[Edge]:
    """Return an acyclic posterior-threshold edge list."""
    return sorted(dag_from_edge_probs(edge_probs, nodes, threshold=threshold).edges())


def _validated_edge_probabilities(
    edge_probs: Mapping[Edge, float], nodes: Sequence[str]
) -> Dict[Edge, float]:
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        raise ValueError("nodes must be unique")
    validated: Dict[Edge, float] = {}
    for (parent, child), raw_probability in edge_probs.items():
        if parent not in node_set or child not in node_set:
            raise ValueError(f"edge {(parent, child)!r} contains an unknown node")
        if parent == child:
            raise ValueError("self-edge probabilities are not valid DAG marginals")
        probability = float(raw_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"edge probability for {(parent, child)!r} must lie in [0, 1]")
        validated[(parent, child)] = probability
    return validated


def _binary64_fraction(value: float) -> Fraction:
    """Return the exact rational value represented by one finite binary64."""

    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _exact_subset_dp_decision(
    candidate_edges: Sequence[Edge],
    utilities: Mapping[Edge, float],
    nodes: Sequence[str],
    max_indegree: int | None,
) -> tuple[tuple[bool, ...], Fraction, bool, int]:
    """Solve the positive-utility DAG decision exactly for a small node set.

    For a fixed topological prefix, the best incoming parent set of the node
    added last is independent of the earlier choices.  Subset dynamic
    programming therefore finds the global optimum in ``O(d 2**d)`` states.
    Objective comparisons use the exact rational values represented by the
    frozen binary64 coefficients.  Among primary ties, the integer incidence
    mask implements the declared rule: prefer inclusion of the earliest
    sorted directed edge.

    The Boolean return after the objective reports whether more than one
    *edge set* (rather than merely more than one topological order) has the
    exact primary optimum.
    """

    ordered_nodes = tuple(sorted(nodes))
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    edge_index = {edge: index for index, edge in enumerate(candidate_edges)}
    exact_utilities = tuple(
        _binary64_fraction(utilities[edge]) for edge in candidate_edges
    )
    lex_bits = tuple(1 << (len(candidate_edges) - 1 - index) for index in range(len(candidate_edges)))
    incoming_by_child: list[list[tuple[int, int]]] = [
        [] for _ in ordered_nodes
    ]
    for index, (parent, child) in enumerate(candidate_edges):
        incoming_by_child[node_index[child]].append((node_index[parent], index))

    state_count = 1 << len(ordered_nodes)
    best_utility: list[Fraction | None] = [None] * state_count
    best_edge_mask: list[int] = [0] * state_count
    has_distinct_tie: list[bool] = [False] * state_count
    best_utility[0] = Fraction(0)
    transitions = 0

    for state in range(1, state_count):
        state_best_utility: Fraction | None = None
        state_best_mask = 0
        state_has_tie = False
        for child_index in range(len(ordered_nodes)):
            child_bit = 1 << child_index
            if not state & child_bit:
                continue
            previous = state ^ child_bit
            previous_utility = best_utility[previous]
            if previous_utility is None:  # pragma: no cover - DP invariant
                continue
            eligible = [
                edge_idx
                for parent_idx, edge_idx in incoming_by_child[child_index]
                if previous & (1 << parent_idx)
            ]
            eligible.sort(key=lambda edge_idx: (-exact_utilities[edge_idx], edge_idx))
            limit = len(eligible) if max_indegree is None else min(max_indegree, len(eligible))
            chosen = eligible[:limit]
            local_utility = sum(
                (exact_utilities[edge_idx] for edge_idx in chosen), Fraction(0)
            )
            local_mask = 0
            for edge_idx in chosen:
                local_mask |= lex_bits[edge_idx]
            local_has_tie = (
                limit > 0
                and limit < len(eligible)
                and exact_utilities[eligible[limit - 1]]
                == exact_utilities[eligible[limit]]
            )
            candidate_utility = previous_utility + local_utility
            candidate_mask = best_edge_mask[previous] | local_mask
            candidate_has_tie = has_distinct_tie[previous] or local_has_tie
            transitions += 1

            if state_best_utility is None or candidate_utility > state_best_utility:
                state_best_utility = candidate_utility
                state_best_mask = candidate_mask
                state_has_tie = candidate_has_tie
            elif candidate_utility == state_best_utility:
                if candidate_mask != state_best_mask:
                    state_has_tie = True
                state_has_tie = state_has_tie or candidate_has_tie
                if candidate_mask > state_best_mask:
                    state_best_mask = candidate_mask

        best_utility[state] = state_best_utility
        best_edge_mask[state] = state_best_mask
        has_distinct_tie[state] = state_has_tie

    full_state = state_count - 1
    optimum = best_utility[full_state]
    if optimum is None:  # pragma: no cover - the empty edge set is feasible
        raise RuntimeError("exact subset-DP DAG decision found no feasible graph")
    mask = best_edge_mask[full_state]
    selected = tuple(bool(mask & bit) for bit in lex_bits)
    return selected, optimum, has_distinct_tie[full_state], transitions


def optimal_bayes_action_dag(
    edge_probs: Mapping[Edge, float],
    nodes: Sequence[str],
    *,
    max_indegree: int | None = None,
    loss: Literal[
        "pair_state_hamming",
        "pair_state_log_probability",
        "directed_edge_hamming",
        "directed_hamming",
        "log_odds",
    ] = "pair_state_hamming",
    time_limit_seconds: float | None = None,
    require_optimal: bool = True,
    lexicographic_epsilon: float = 0.0,
    probability_clip: float = 1e-12,
    milp_objective_audit_tolerance: float = 1e-9,
) -> nx.DiGraph:
    """Return the globally constrained marginal-probability DAG.

    The default decision is the Bayes action under additive *pair-state*
    Hamming loss, where each unordered pair is in one of three mutually
    exclusive states: no edge, ``i -> j``, or ``j -> i``.  A missing/extra
    adjacency and an orientation reversal therefore both cost one, matching
    the paper's directed pair-state SHD.  Relative to choosing no edge, the
    utility of ``i -> j`` is ``P(i -> j) - P(no edge)``.

    ``loss='pair_state_log_probability'`` instead maximizes the product of the
    three-state marginal probabilities (equivalently, their clipped log
    probabilities) under the graph constraints.  It is a useful sensitivity
    summary but is not a Bayes action for SHD.  ``directed_edge_hamming``
    treats the two directed-edge indicators independently, so a reversal costs
    two; ``directed_hamming`` is retained as its backwards-compatible alias.
    ``log_odds`` retains the older independent-edge log-odds heuristic, with
    the same declared clipping constant used to keep boundary probabilities
    finite for the optimizer.

    A mixed-integer program jointly chooses directed edges and continuous
    topological ranks. The strict rank constraint makes every feasible result
    acyclic, and optional per-child constraints enforce ``max_indegree``. The
    returned graph records solver status, objective semantics, and optimality
    information in ``graph.graph``.
    """

    ordered_nodes = tuple(str(node) for node in nodes)
    if any(str(node) != node for node in nodes):
        raise TypeError("nodes must be strings")
    if max_indegree is not None and (
        isinstance(max_indegree, bool)
        or not isinstance(max_indegree, int)
        or max_indegree < 0
    ):
        raise ValueError("max_indegree must be a non-negative integer or None")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, (int, float))
        or not math.isfinite(float(time_limit_seconds))
        or time_limit_seconds <= 0
    ):
        raise ValueError("time_limit_seconds must be positive")
    if (
        isinstance(lexicographic_epsilon, bool)
        or not isinstance(lexicographic_epsilon, (int, float))
        or not math.isfinite(float(lexicographic_epsilon))
        or lexicographic_epsilon < 0
    ):
        raise ValueError("lexicographic_epsilon must be non-negative")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must lie in (0, 0.5)")
    if (
        isinstance(milp_objective_audit_tolerance, bool)
        or not isinstance(milp_objective_audit_tolerance, (int, float))
        or not math.isfinite(float(milp_objective_audit_tolerance))
        or milp_objective_audit_tolerance < 0
    ):
        raise ValueError("milp_objective_audit_tolerance must be non-negative")
    allowed_losses = {
        "pair_state_hamming",
        "pair_state_log_probability",
        "directed_edge_hamming",
        "directed_hamming",
        "log_odds",
    }
    if loss not in allowed_losses:
        raise ValueError(f"unsupported loss: {loss}")

    probabilities = _validated_edge_probabilities(edge_probs, ordered_nodes)
    pair_probabilities: Dict[FrozenSet[str], Tuple[float, float, float]] = {}
    lexical_nodes = tuple(sorted(ordered_nodes))
    for left_index, left in enumerate(lexical_nodes):
        for right in lexical_nodes[left_index + 1 :]:
            forward = probabilities.get((left, right), 0.0)
            reverse = probabilities.get((right, left), 0.0)
            if forward + reverse > 1.0 + 1e-10:
                raise ValueError(
                    "opposite edge probabilities must sum to at most one for "
                    f"{(left, right)!r}; got {forward + reverse}"
                )
            no_edge = max(0.0, 1.0 - forward - reverse)
            pair_probabilities[frozenset((left, right))] = (
                forward,
                reverse,
                no_edge,
            )

    utilities: Dict[Edge, float] = {}
    for edge, probability in probabilities.items():
        parent, child = edge
        left, right = sorted((parent, child))
        _, _, no_edge = pair_probabilities[frozenset((left, right))]
        # ``probability`` already corresponds to the directed edge being
        # scored. Do not reinterpret it through caller node order: node order
        # need not be lexicographic.
        orientation_probability = probability
        if loss == "pair_state_hamming":
            utility = orientation_probability - no_edge
        elif loss == "pair_state_log_probability":
            clipped_orientation = float(
                np.clip(orientation_probability, probability_clip, 1.0 - probability_clip)
            )
            clipped_no_edge = float(
                np.clip(no_edge, probability_clip, 1.0 - probability_clip)
            )
            utility = math.log(clipped_orientation) - math.log(clipped_no_edge)
        elif loss in ("directed_edge_hamming", "directed_hamming"):
            utility = 2.0 * probability - 1.0
        elif loss == "log_odds":
            clipped_probability = float(
                np.clip(probability, probability_clip, 1.0 - probability_clip)
            )
            utility = math.log(clipped_probability) - math.log1p(
                -clipped_probability
            )
        else:  # pragma: no cover - Literal protects typed callers
            raise ValueError(f"unsupported loss: {loss}")
        # Non-positive edges can never improve an additive objective and only
        # consume indegree/acyclicity capacity, so removing them is exact.
        if utility > 0.0:
            utilities[edge] = utility

    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_nodes)
    graph.graph.update(
        {
            "summary_method": (
                "globally_constrained_bayes_action"
                if loss == "pair_state_hamming"
                else "globally_constrained_marginal_utility"
            ),
            "loss": loss,
            "objective_semantics": {
                "pair_state_hamming": "posterior expected pair-state SHD reduction",
                "pair_state_log_probability": "clipped product of pair-state marginals",
                "directed_edge_hamming": "posterior expected directed-edge Hamming reduction",
                "directed_hamming": "posterior expected directed-edge Hamming reduction",
                "log_odds": "independent directed-edge log-odds heuristic",
            }[loss],
            "max_indegree": max_indegree,
            "pair_state_semantics": "none,forward,reverse are mutually exclusive",
            "probability_clip": float(probability_clip),
            "solver": "scipy.optimize.milp/HiGHS",
            "solver_interface": "scipy.optimize.milp",
            "scipy_version": scipy.__version__,
            "solver_backend": "HiGHS",
            "solver_backend_version": _HIGHS_VERSION,
            # Backwards-compatible name; unlike the previous implementation,
            # this is the backend version rather than the SciPy interface
            # package version.
            "solver_version": _HIGHS_VERSION,
            "time_limit_seconds": time_limit_seconds,
            "require_optimal": bool(require_optimal),
            "milp_objective_audit_tolerance": float(
                milp_objective_audit_tolerance
            ),
            "milp_objective_scale_exponent": _MILP_OBJECTIVE_SCALE_EXPONENT,
            "objective_scaling": "uniform exact power of two; audits in original units",
            "tie_break": (
                "exclude utilities <= 0, then exact lexicographic inclusion over "
                "sorted positive-utility directed-edge variables"
                if lexicographic_epsilon == 0.0
                else "explicit rank perturbation; inspect lexicographic_epsilon"
            ),
            "decision_domain_policy": (
                "non-positive-utility edges are deterministically excluded before "
                "optimization; MILP uniqueness and tie evidence refer to the "
                "strictly-positive-utility reduced variable set"
            ),
        }
    )
    if not utilities:
        graph.graph.update(
            {
                "solver_status": 0,
                "solver_message": "empty graph is optimal; no edge has positive utility",
                "primary_solver_status": 0,
                "primary_solver_message": (
                    "empty graph is optimal; no edge has positive utility"
                ),
                "tie_solver_status": None,
                "optimal": True,
                "mip_gap": 0.0,
                "objective_utility": 0.0,
                "candidate_positive_utility_edges": 0,
                "lexicographic_epsilon": float(lexicographic_epsilon),
                "tie_detected": False,
                "tie_break_certified": True,
                "primary_optimum_unique": True,
                "primary_optimum_unique_scope": (
                    "strictly-positive-utility reduced variable set"
                ),
                "decision_unique_after_tie_break": True,
                "lexicographic_subproblems": 0,
                "time_limit_scope": "per MILP subproblem",
                "objective_equality_rule": (
                    "exact equality of binary64 coefficients as rational numbers"
                ),
                "primary_objective_binary64_rational": "0/1",
                "selected_objective_binary64_rational": "0/1",
                "primary_vs_selected_regret": 0.0,
                "primary_vs_selected_regret_binary64_rational": "0/1",
                "milp_primary_vs_certified_regret": 0.0,
                "milp_primary_vs_certified_regret_binary64_rational": "0/1",
                "exact_primary_certification": {
                    "applied": True,
                    "backend": "trivial_empty_positive_utility_domain",
                    "status": "certified",
                    "state_transition_count": 0,
                },
            }
        )
        return graph

    candidate_edges = sorted(utilities)
    edge_count = len(candidate_edges)
    node_count = len(ordered_nodes)
    node_index = {node: idx for idx, node in enumerate(ordered_nodes)}
    variable_count = edge_count + node_count

    # scipy.optimize.milp minimizes. By default the scientific objective is
    # unperturbed. ``lexicographic_epsilon`` is an explicit legacy sensitivity,
    # never a hidden part of the confirmatory objective.
    objective = np.zeros(variable_count, dtype=float)
    for edge_idx, edge in enumerate(candidate_edges):
        tie_penalty = lexicographic_epsilon * float(edge_idx + 1) / float(edge_count + 1)
        objective[edge_idx] = -float(utilities[edge]) + tie_penalty

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    upper_bounds[edge_count:] = max(0, node_count - 1)
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:edge_count] = 1

    row_indices: List[int] = []
    column_indices: List[int] = []
    values: List[float] = []
    constraint_lower: List[float] = []
    constraint_upper: List[float] = []

    # If x_uv = 1, rank(u) + 1 <= rank(v). For x_uv = 0 the
    # constraint is vacuous over ranks in [0, node_count - 1].
    for edge_idx, (parent, child) in enumerate(candidate_edges):
        row = len(constraint_upper)
        row_indices.extend((row, row, row))
        column_indices.extend(
            (edge_idx, edge_count + node_index[parent], edge_count + node_index[child])
        )
        values.extend((float(node_count), 1.0, -1.0))
        constraint_lower.append(-np.inf)
        constraint_upper.append(float(node_count - 1))

    if max_indegree is not None:
        for child in ordered_nodes:
            incoming = [
                edge_idx
                for edge_idx, (_, candidate_child) in enumerate(candidate_edges)
                if candidate_child == child
            ]
            if not incoming:
                continue
            row = len(constraint_upper)
            row_indices.extend([row] * len(incoming))
            column_indices.extend(incoming)
            values.extend([1.0] * len(incoming))
            constraint_lower.append(-np.inf)
            constraint_upper.append(float(max_indegree))

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_upper), variable_count),
    ).tocsr()
    # SciPy does not list ``mip_abs_gap`` in its narrow public options type,
    # but forwards recognized HiGHS options verbatim.  HiGHS otherwise uses
    # an absolute MIP gap of 1e-6, which can label a binary solution optimal
    # even when the exact binary64-rational audit finds a tiny improvement.
    options: Dict[str, float | bool] = {
        "presolve": True,
        "mip_rel_gap": 0.0,
        "mip_abs_gap": 0.0,
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
        "mip_feasibility_tolerance": 1e-9,
    }
    if time_limit_seconds is not None:
        options["time_limit"] = float(time_limit_seconds)
    base_lower = np.asarray(constraint_lower, dtype=float)
    base_upper = np.asarray(constraint_upper, dtype=float)

    def solve_milp(
        solve_objective: np.ndarray,
        extra_rows: Sequence[np.ndarray] = (),
        extra_lower: Sequence[float] = (),
        extra_upper: Sequence[float] = (),
    ):
        if extra_rows:
            extra_matrix = coo_matrix(np.asarray(extra_rows, dtype=float)).tocsr()
            solve_matrix = vstack((matrix, extra_matrix), format="csr")
            solve_lower = np.concatenate((base_lower, np.asarray(extra_lower, dtype=float)))
            solve_upper = np.concatenate((base_upper, np.asarray(extra_upper, dtype=float)))
        else:
            solve_matrix = matrix
            solve_lower = base_lower
            solve_upper = base_upper
        # SciPy warns because these backend-native tolerances are outside its
        # small documented option subset, even though it explicitly forwards
        # them to HiGHS.  Suppress only that known forwarding warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Unrecognized options detected: .*",
                category=RuntimeWarning,
            )
            scaled_objective = np.ldexp(solve_objective, _MILP_OBJECTIVE_SCALE_EXPONENT)
            if not np.isfinite(scaled_objective).all():
                raise ValueError("MILP objective overflows under power-of-two scaling")
            answer = milp(
                c=scaled_objective,
                integrality=integrality,
                bounds=Bounds(lower_bounds, upper_bounds),
                constraints=LinearConstraint(solve_matrix, solve_lower, solve_upper),
                options=options,
            )
            # Restore diagnostics to the declared scientific objective units.
            # The relative MIP gap is unchanged by uniform positive scaling.
            for field in ("fun", "mip_dual_bound"):
                value = getattr(answer, field, None)
                if value is not None:
                    answer[field] = float(np.ldexp(value, -_MILP_OBJECTIVE_SCALE_EXPONENT))
            return answer

    result = solve_milp(objective)
    optimal = int(result.status) == 0
    if result.x is None or (require_optimal and not optimal):
        raise RuntimeError(
            "constrained marginal-DAG optimization did not produce the required optimum: "
            f"status={result.status}, message={result.message}"
        )

    # HiGHS is deterministic for a fixed build and input, but its feasibility
    # tolerances are not a valid mathematical definition of an objective tie.
    # Every equality and regret check below therefore uses exact rational sums
    # of the binary64 coefficients.  For the confirmatory d<=10 domain, an
    # independent subset-DP oracle certifies both the primary optimum and the
    # unique lexicographic Bayes action.  Larger exploratory domains retain the
    # MILP but never replace its primary solution by an approximately-equal,
    # strictly lower-utility solution.
    primary_result = result
    primary_optimal = optimal
    milp_primary_selected = np.asarray(
        primary_result.x[:edge_count] >= 0.5, dtype=bool
    )
    exact_coefficients = tuple(
        _binary64_fraction(utilities[edge]) for edge in candidate_edges
    )

    def exact_selected_utility(selected: Sequence[bool]) -> Fraction:
        return sum(
            (
                exact_coefficients[edge_idx]
                for edge_idx in range(edge_count)
                if bool(selected[edge_idx])
            ),
            Fraction(0),
        )

    milp_primary_exact_utility = exact_selected_utility(milp_primary_selected)
    certified_primary_exact_utility = milp_primary_exact_utility
    final_selected = np.asarray(milp_primary_selected, dtype=bool)
    tie_detected = False
    tie_break_certified = lexicographic_epsilon == 0.0 and primary_optimal
    lexicographic_subproblems = 0
    tie_solver_status: int | None = None
    exact_primary_certification: dict[str, object] = {
        "applied": False,
        "backend": "HiGHS plus exact binary64-rational candidate audit",
        "status": "not_applicable_explicit_objective_perturbation"
        if lexicographic_epsilon != 0.0
        else "candidate_audited",
        "state_transition_count": 0,
    }
    milp_primary_vs_certified_regret = Fraction(0)

    if lexicographic_epsilon == 0.0 and primary_optimal:
        if node_count <= _EXACT_SUBSET_DP_MAX_NODES:
            (
                exact_selected,
                exact_optimum,
                exact_has_tie,
                exact_transitions,
            ) = _exact_subset_dp_decision(
                candidate_edges,
                utilities,
                ordered_nodes,
                max_indegree,
            )
            certified_primary_exact_utility = exact_optimum
            milp_regret = exact_optimum - milp_primary_exact_utility
            if milp_regret < 0:
                raise RuntimeError(
                    "HiGHS primary DAG exceeds the exact subset-DP optimum; "
                    "the independent certifier is inconsistent: "
                    f"exact_regret={milp_regret}"
                )
            milp_primary_vs_certified_regret = milp_regret
            final_selected = np.asarray(exact_selected, dtype=bool)
            tie_detected = bool(exact_has_tie)
            tie_solver_status = 0 if tie_detected else None
            exact_primary_certification = {
                "applied": True,
                "backend": "exact_binary64_rational_subset_dynamic_programming",
                "status": "certified",
                "node_limit": _EXACT_SUBSET_DP_MAX_NODES,
                "state_transition_count": exact_transitions,
            }
        else:
            # Exclude precisely the primary edge set. Ranks are intentionally
            # absent: different topological ranks for one DAG are not a
            # decision tie.
            no_good = np.zeros(variable_count, dtype=float)
            no_good[:edge_count] = np.where(milp_primary_selected, -1.0, 1.0)
            no_good_lower = 1.0 - float(np.sum(milp_primary_selected))
            alternative = solve_milp(
                objective,
                (no_good,),
                (no_good_lower,),
                (np.inf,),
            )
            lexicographic_subproblems += 1
            if int(alternative.status) == 0 and alternative.x is not None:
                alternative_selected = alternative.x[:edge_count] >= 0.5
                alternative_exact_utility = exact_selected_utility(
                    alternative_selected
                )
                if alternative_exact_utility > milp_primary_exact_utility:
                    audit_regret = (
                        alternative_exact_utility - milp_primary_exact_utility
                    )
                    if float(audit_regret) > milp_objective_audit_tolerance:
                        raise RuntimeError(
                            "HiGHS uniqueness audit found a materially better exact "
                            "binary64-rational objective than its primary result: "
                            f"regret={audit_regret}, "
                            f"float_regret={float(audit_regret):.17g}, "
                            f"audit_tolerance={milp_objective_audit_tolerance:.17g}, "
                            f"primary_edges={int(np.sum(milp_primary_selected))}, "
                            f"alternative_edges={int(np.sum(alternative_selected))}"
                        )
                    # HiGHS optimizes in floating arithmetic.  When its
                    # excluded-incumbent audit exposes only a sub-tolerance
                    # discrepancy, retain the strictly better graph rather
                    # than silently accepting the first incumbent.  This does
                    # not redefine the objective: both utilities are still
                    # compared as exact sums of the input binary64 values.
                    final_selected = np.asarray(alternative_selected, dtype=bool)
                    certified_primary_exact_utility = alternative_exact_utility
                    milp_primary_vs_certified_regret = audit_regret
                    tie_break_certified = False
                    exact_primary_certification = {
                        "applied": False,
                        "backend": "HiGHS plus exact binary64-rational candidate audit",
                        "status": "sub_tolerance_incumbent_refined",
                        "state_transition_count": 0,
                        "observed_regret": float(audit_regret),
                        "audit_tolerance": float(milp_objective_audit_tolerance),
                    }
                tie_detected = (
                    alternative_exact_utility == milp_primary_exact_utility
                )
            elif int(alternative.status) == 2:
                tie_detected = False
            else:
                tie_break_certified = False
                if require_optimal:
                    raise RuntimeError(
                        "could not certify uniqueness of the constrained marginal-DAG optimum: "
                        f"status={alternative.status}, message={alternative.message}"
                    )

            if tie_detected:
                # Keep a known primary-optimal incumbent throughout.  To test
                # whether the next lexical edge can be included, re-optimize
                # the *primary* objective under that forced decision and
                # accept the candidate only when its exact rational utility is
                # equal.  We never use a floating equality band as a tie rule.
                fixed_rows: list[np.ndarray] = []
                fixed_values: list[float] = []
                incumbent = np.asarray(milp_primary_selected, dtype=bool)
                for edge_idx in range(edge_count):
                    if incumbent[edge_idx]:
                        fixed_value = 1.0
                    else:
                        forced_row = np.zeros(variable_count, dtype=float)
                        forced_row[edge_idx] = 1.0
                        trial = solve_milp(
                            objective,
                            (*fixed_rows, forced_row),
                            (*fixed_values, 1.0),
                            (*fixed_values, 1.0),
                        )
                        lexicographic_subproblems += 1
                        if int(trial.status) == 0 and trial.x is not None:
                            trial_selected = trial.x[:edge_count] >= 0.5
                            trial_exact_utility = exact_selected_utility(
                                trial_selected
                            )
                            if trial_exact_utility > certified_primary_exact_utility:
                                audit_regret = (
                                    trial_exact_utility
                                    - certified_primary_exact_utility
                                )
                                if float(audit_regret) > milp_objective_audit_tolerance:
                                    raise RuntimeError(
                                        "lexicographic audit found a materially better "
                                        "exact binary64-rational objective than the "
                                        "audited incumbent: "
                                        f"regret={audit_regret}, "
                                        f"float_regret={float(audit_regret):.17g}, "
                                        f"audit_tolerance={milp_objective_audit_tolerance:.17g}, "
                                        f"edge_index={edge_idx}"
                                    )
                                # Apply the same policy as the excluded-incumbent
                                # audit above. A forced lexical subproblem can
                                # expose a sub-tolerance improvement that HiGHS
                                # did not distinguish in the primary solve. Keep
                                # the strictly better exact-binary64 incumbent,
                                # but do not claim an exact optimality/tie-break
                                # certificate for this large-domain MILP.
                                incumbent = np.asarray(trial_selected, dtype=bool)
                                certified_primary_exact_utility = trial_exact_utility
                                milp_primary_vs_certified_regret = (
                                    trial_exact_utility
                                    - milp_primary_exact_utility
                                )
                                tie_detected = False
                                tie_break_certified = False
                                fixed_value = 1.0
                                tie_solver_status = int(trial.status)
                                exact_primary_certification = {
                                    "applied": False,
                                    "backend": (
                                        "HiGHS plus exact binary64-rational "
                                        "candidate audit"
                                    ),
                                    "status": (
                                        "sub_tolerance_lexicographic_"
                                        "incumbent_refined"
                                    ),
                                    "state_transition_count": 0,
                                    "observed_regret": float(
                                        milp_primary_vs_certified_regret
                                    ),
                                    "audit_tolerance": float(
                                        milp_objective_audit_tolerance
                                    ),
                                    "triggering_edge_index": edge_idx,
                                }
                            elif trial_exact_utility == certified_primary_exact_utility:
                                incumbent = np.asarray(trial_selected, dtype=bool)
                                tie_detected = True
                                fixed_value = 1.0
                                tie_solver_status = int(trial.status)
                            else:
                                fixed_value = 0.0
                        elif int(trial.status) == 2:
                            fixed_value = 0.0
                        else:
                            tie_break_certified = False
                            if require_optimal:
                                raise RuntimeError(
                                    "could not certify the lexicographic constrained-DAG tie break: "
                                    f"status={trial.status}, message={trial.message}"
                                )
                            break
                    fixed_row = np.zeros(variable_count, dtype=float)
                    fixed_row[edge_idx] = 1.0
                    fixed_rows.append(fixed_row)
                    fixed_values.append(fixed_value)
                # Even when a sub-tolerance refinement prevents an exact
                # certificate, ``incumbent`` is the best exact-binary64 graph
                # encountered and is never worse than the primary result.
                final_selected = incumbent

    selected_exact_utility = exact_selected_utility(final_selected)
    primary_vs_selected_regret = (
        certified_primary_exact_utility - selected_exact_utility
    )
    if lexicographic_epsilon == 0.0 and primary_optimal and primary_vs_selected_regret != 0:
        raise RuntimeError(
            "deterministic DAG tie handling changed the exact primary objective; "
            f"regret={primary_vs_selected_regret}"
        )

    selected_edges = [
        edge
        for edge_idx, edge in enumerate(candidate_edges)
        if final_selected[edge_idx]
    ]
    graph.add_edges_from(
        (
            parent,
            child,
            {
                "posterior_probability": probabilities[(parent, child)],
                "decision_utility": utilities[(parent, child)],
            },
        )
        for parent, child in selected_edges
    )
    if not nx.is_directed_acyclic_graph(graph):  # pragma: no cover - solver invariant
        raise RuntimeError("MILP returned a cyclic graph despite topological-rank constraints")
    if max_indegree is not None and max(dict(graph.in_degree()).values(), default=0) > max_indegree:
        raise RuntimeError("MILP returned a graph violating the indegree constraint")

    graph.graph.update(
        {
            "solver_status": int(primary_result.status),
            "solver_message": str(primary_result.message),
            "optimal": primary_optimal,
            "mip_gap": (
                None
                if getattr(primary_result, "mip_gap", None) is None
                else float(primary_result.mip_gap)
            ),
            "primary_solver_status": int(primary_result.status),
            "primary_solver_message": str(primary_result.message),
            "tie_solver_status": tie_solver_status,
            "objective_utility": float(sum(utilities[edge] for edge in selected_edges)),
            "candidate_positive_utility_edges": edge_count,
            "lexicographic_epsilon": float(lexicographic_epsilon),
            "tie_detected": bool(tie_detected),
            "tie_break_certified": bool(tie_break_certified),
            "primary_optimum_unique": bool(
                not tie_detected and tie_break_certified
            ),
            "primary_optimum_unique_scope": (
                "strictly-positive-utility reduced variable set"
            ),
            "decision_unique_after_tie_break": bool(tie_break_certified),
            "lexicographic_subproblems": int(lexicographic_subproblems),
            "time_limit_scope": "per MILP subproblem",
            "objective_equality_rule": (
                "exact equality of binary64 coefficients as rational numbers; "
                "no absolute or relative near-tie tolerance"
            ),
            "primary_objective_binary64_rational": (
                f"{certified_primary_exact_utility.numerator}/"
                f"{certified_primary_exact_utility.denominator}"
            ),
            "selected_objective_binary64_rational": (
                f"{selected_exact_utility.numerator}/"
                f"{selected_exact_utility.denominator}"
            ),
            "primary_vs_selected_regret": float(primary_vs_selected_regret),
            "primary_vs_selected_regret_binary64_rational": (
                f"{primary_vs_selected_regret.numerator}/"
                f"{primary_vs_selected_regret.denominator}"
            ),
            "milp_primary_vs_certified_regret": float(
                milp_primary_vs_certified_regret
            ),
            "milp_primary_vs_certified_regret_binary64_rational": (
                f"{milp_primary_vs_certified_regret.numerator}/"
                f"{milp_primary_vs_certified_regret.denominator}"
            ),
            "exact_primary_certification": exact_primary_certification,
        }
    )
    return graph


def _ordered_dag_edges(dag: nx.DiGraph) -> List[Edge]:
    """Chickering's deterministic edge order for compelled-edge labelling."""

    topo = list(nx.lexicographical_topological_sort(dag, key=lambda node: str(node)))
    topo_index = {node: index for index, node in enumerate(topo)}
    remaining = set(dag.edges())
    ordered: List[Edge] = []
    while remaining:
        children = {child for _, child in remaining}
        child = min(children, key=lambda node: topo_index[node])
        parents = [parent for parent, target in remaining if target == child]
        parent = max(parents, key=lambda node: topo_index[node])
        edge = (parent, child)
        ordered.append((str(parent), str(child)))
        remaining.remove(edge)
    return ordered


def dag_to_cpdag(dag: nx.DiGraph) -> CPDAG:
    """Convert a DAG to its completed PDAG using Chickering's algorithm.

    Directed output edges are compelled in every Markov-equivalent DAG;
    undirected output edges are reversible. Node and edge ordering is made
    deterministic with lexicographical topological sorting.
    """

    if not isinstance(dag, nx.DiGraph) or isinstance(dag, nx.MultiDiGraph):
        raise TypeError("dag must be a simple networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("dag_to_cpdag requires a directed acyclic graph")
    if any(parent == child for parent, child in dag.edges()):
        raise ValueError("self edges are not valid in a DAG")

    # Work with stable string labels, rejecting ambiguous coercions.
    nodes = tuple(sorted((str(node) for node in dag.nodes())))
    if len(set(nodes)) != len(dag.nodes()) or any(str(node) != node for node in dag.nodes()):
        raise TypeError("DAG nodes must be unique strings")
    ordered_edges = _ordered_dag_edges(dag)
    labels: Dict[Edge, Literal["unknown", "compelled", "reversible"]] = {
        edge: "unknown" for edge in ordered_edges
    }
    parents = {str(node): {str(parent) for parent in dag.predecessors(node)} for node in dag.nodes()}

    for edge in ordered_edges:
        if labels[edge] != "unknown":
            continue
        parent, child = edge
        restart = False
        compelled_into_parent = sorted(
            source
            for source in parents[parent]
            if labels.get((source, parent)) == "compelled"
        )
        for source in compelled_into_parent:
            if source not in parents[child]:
                labels[edge] = "compelled"
                for incoming_parent in parents[child]:
                    labels[(incoming_parent, child)] = "compelled"
                restart = True
                break
            labels[(source, child)] = "compelled"
        if restart:
            continue

        creates_unshielded_collider = any(
            other_parent != parent and other_parent not in parents[parent]
            for other_parent in parents[child]
        )
        if creates_unshielded_collider:
            labels[edge] = "compelled"
            for incoming_parent in parents[child]:
                if labels[(incoming_parent, child)] == "unknown":
                    labels[(incoming_parent, child)] = "compelled"
        else:
            labels[edge] = "reversible"
            for incoming_parent in parents[child]:
                if labels[(incoming_parent, child)] == "unknown":
                    labels[(incoming_parent, child)] = "reversible"

    if any(label == "unknown" for label in labels.values()):  # pragma: no cover - algorithm invariant
        raise RuntimeError("CPDAG conversion left an edge unlabelled")
    directed = frozenset(edge for edge, label in labels.items() if label == "compelled")
    undirected = frozenset(
        frozenset(edge) for edge, label in labels.items() if label == "reversible"
    )
    return CPDAG(nodes=nodes, directed_edges=directed, undirected_edges=undirected)
