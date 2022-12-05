import pytest
from bs4 import BeautifulSoup
from evaluation import RankingGraph, EDRC, convert_ranking_to_graph


@pytest.fixture
def graph():
    return RankingGraph()


def test_insert_pair(graph):
    graph.insert((1, 2), 1)
    assert graph.vertices == {1, 2}, f'{graph.vertices}'


def test_edge_direction_for_first_relevant(graph):
    graph.insert((1, 2), 1)
    assert graph.edges == {(1, 2)}, f'{graph.edges}'


def test_edge_direction_for_second_relevant(graph):
    graph.insert((1, 2), 2)
    assert graph.edges == {(2, 1)}, f'{graph.edges}'


def test_insert_for_two_relevant_nodes(graph):
    for edge in [('a', 'b'), ('a', 'd')]:
        graph.insert(edge, 4)
    assert graph.relevant == {'a', 'b', 'd'}

# we no longer check for that automatically since the actual annotations are not
# necessarily constistent?
def test_inconsistent_graph(graph):
    graph.continuous_consistency_check = True
    graph.insert((1, 2), 4)
    graph.insert((2, 3), 4)
    with pytest.raises(Exception):
        graph.insert((3, 1), 1)


def test_conflicting_pair(graph):
    pairs = [(1, 2), (2, 3), (1, 2)]
    labels = [1, 2, 2]
    with pytest.raises(Exception):
        for pair, label in zip(pairs, labels):
            graph.insert(pair, label)


def test_conflict_for_irrelevant_node(graph):
    pairs = [(2, 3), (2, 4), (1, 2)]
    labels = [2, 1, 0]
    with pytest.raises(Exception):
        for pair, label in zip(pairs, labels):
            graph.insert(pair, label)


def test_duplicate_shuffled_pair(graph):
    pairs = [(1, 2), (2, 3), (2, 1)]
    labels = [1, 2, 1]
    with pytest.raises(Exception):
        for pair, label in zip(pairs, labels):
            graph.insert(pair, label)


def test_shared_dependency_for_same_rank_pair(graph):
    pairs = [(1, 2), (2, 3), (4, 1), (4, 5)]
    labels = [4, 4, 3, 5]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    assert graph.edges_by_innode[1] == graph.edges_by_innode[4]\
        and graph.edges_by_outnode[1] == graph.edges_by_outnode[4]


def test_hierarchy(graph):
    pairs = [(1, 2), (2, 3), (1, 4)]
    labels = [4, 4, 4]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    assert graph.hierarchies() == [(1, 2, 3), (1, 4)]


def test_hierarchy_for_disordered_insert(graph):
    pairs = [(2, 4), (1, 4), (2, 3), (5, 4)]
    labels = [5, 3, 1, 4]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    assert graph.hierarchies() == [(5, 1, 2, 3), (5, 4, 2, 3)]


@pytest.mark.skip('not needed in current application scenario')
def test_explicit_agreement_for_single_chain_insert(graph):
    pairs = [(1, 2), (2, 3)]
    labels = [1, 1]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    system_ranking = [1, 3, 2]
    assert graph.count_agreements(system_ranking, True) == 1


@pytest.mark.skip('not needed in current application scenario')
def test_implicit_agreement_for_single_chain_insert(graph):
    pairs = [(1, 2), (2, 3)]
    labels = [1, 1]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    system_ranking = [1, 3, 2]
    assert graph.count_agreements(system_ranking) == 2


@pytest.mark.skip('not needed in current application scenario')
def test_implicit_agreement_for_single_relevant_and_multiple_irrelevant(graph):
    pairs = [(1, 2), (2, 3), (3, 4)]
    labels = [1, 0, 0]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    system_ranking = [1, 3, 2, 4]
    assert graph.count_agreements(system_ranking) == 4


def test_implicit_precision(graph):
    pairs = [(1, 2), (2, 3), (2, 4)]
    labels = [1, 2, 0]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    system_ranking = [1, 3, 2, 4]
    assert graph.precision(system_ranking) == 5/6


def test_explicit_precision(graph):
    pairs = [(1, 2), (2, 3), (2, 4)]
    labels = [1, 2, 0]
    for pair, label in zip(pairs, labels):
        graph.insert(pair, label)
    system_ranking = [1, 3, 2, 4]
    assert graph.precision(system_ranking, True) == 1.0


def test_paper_example():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'd'), ('a', 'e'),
                 ('c', 'd'), ('b', 'd'), ('b', 'e')]:
        gold.insert(edge, 4)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('c', 'a'), ('c', 'b'), ('c', 'd'),
                 ('a', 'e'), ('b', 'e'), ('d', 'e')]:
        pred.insert(edge, 4)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert round(result, 4) == round(5/29, 4),\
        f'EDRC should be {5/29} not {result}'


def test_paper_example_with_equally_ranked_vertices():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'd'), ('a', 'e'),
                 ('c', 'd'), ('b', 'd'), ('b', 'e')]:
        gold.insert(edge, 4)
    gold.insert(('a', 'b'), 3)
    gold.insert(('c', 'e'), 3)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('c', 'a'), ('c', 'b'), ('c', 'd'),
                 ('a', 'e'), ('b', 'e'), ('d', 'e')]:
        pred.insert(edge, 4)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert result == 0


def test_edrc_perfect_correlation_complete_graph():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b')]:
        gold.insert(edge, 4)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b')]:
        pred.insert(edge, 4)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert result == 1.0


def test_edrc_perfect_negative_correlation_complete_graph():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b')]:
        gold.insert(edge, 4)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b')]:
        pred.insert(edge, 5)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert result == -1.0


def test_edrc_perfect_correlation():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b'),
                 ('c', 'd')]:
        gold.insert(edge, 4)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b'),
                 ('c', 'd')]:
        pred.insert(edge, 4)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert round(result, 2) == 0.60


def test_edrc_perfect_negative_correlation():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b'),
                 ('c', 'd')]:
        gold.insert(edge, 4)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'c'), ('a', 'b'), ('c', 'b'),
                 ('c', 'd')]:
        pred.insert(edge, 5)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert round(result, 2) == -0.60


def test_ranking_to_graph():
    ranking = BeautifulSoup(
        """<section><h2>prompt1</h2>
<li>
<p>left<b>target2</b>right</p>
<p>left<b>target1</b>right</p>
<p>left<b>target4</b>right</p>
<p>left<b>target3</b>right</p>
<p>left<b>target5</b>right</p>
</li></section>""",
        features='lxml')('p')
    sents2ids = {('left', 'target1', 'right'): 1,
                 ('left', 'target4', 'right'): 2,
                 ('left', 'target5', 'right'): 3}
    graph = convert_ranking_to_graph(ranking, sents2ids)
    assert graph.edges == {(1, 2), (1, 3), (2, 3)}


def test_infer_ranking(graph):
    for edge in [('a', 'b'), ('a', 'd'), ('c', 'd')]:
        graph.insert(edge, 1)
    graph.infer_ranking()
    assert graph.edges == {('a', 'b'), ('a', 'd'), ('c', 'b'), ('c', 'd')}


def test_paper_example_infering_more_edges():
    gold = RankingGraph()
    for edge in [('a', 'c'), ('a', 'e'),
                 ('b', 'e')]:
        gold.insert(edge, 4)
    for edge in [('a', 'd'), ('c', 'd'), ('b', 'd')]:
        gold.insert(edge, 1)
    gold.infer_ranking()
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'c'), ('a', 'e'),
                 ('b', 'e')]:
        pred.insert(edge, 4)
    for edge in [('a', 'd'), ('c', 'd'),
                 ('b', 'd'), ('e', 'd')]:
        pred.insert(edge, 1)
    edrc = EDRC(gold, pred)
    result = edrc.edrc()
    assert round(result, 4) == 0.6538,\
        f'EDRC should be 0.6538 not {result}'
