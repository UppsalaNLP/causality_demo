import pytest
from bs4 import BeautifulSoup
from binary_evaluation import table, text, join_text_and_labels,\
    retrieve_relevance_judgements_per_sentence, precision_at_k,\
    average_precision, mean_average_precision, prepare_annotations


def test_joined_text_and_labels():
    pairs, _ = join_text_and_labels(text[0:1], table.iloc[:10, :2])
    assert pairs == {'Verkan: “växthuseffekt”':
                     [(4, (0, 1)), (1, (0, 2)), (2, (3, 4)),
                      (0, (5, 6)), (0, (3, 7)), (1, (1, 7)),
                      (2, (5, 8)), (5, (1, 9)), (0, (10, 11)),
                      (0, (12, 2))]}


def test_text_and_labels_for_unrelated_input():
    assert join_text_and_labels(text[0:1], table.iloc[:10, 5:6]) == (
        {}, {'Orsak: “klimatanpassning”': 'Orsak: “klimatanpassning”'})


def test_relevance_judgements_per_sent():
    text_label_pairs = {'prompt':
                        [(1, (15, 2)), (5, (3, 10)), (0, (1, 5))]}
    out = retrieve_relevance_judgements_per_sentence(text_label_pairs)
    assert out['prompt']['relevant'] == [15, 3, 10] and\
        out['prompt']['irrelevant'] == [2, 1, 5]


def test_inconsistent_relevance_judgements_per_sent():
    text_label_pairs = {'prompt':
                        [(1, (15, 2)), (5, (3, 10)), (0, (3, 5))]}
    with pytest.raises(Exception):
        retrieve_relevance_judgements_per_sentence(text_label_pairs)


def test_precision_at_k_for_single_query():
    precision_at_4 = precision_at_k(['sent 1', 'sent 2', 'sent 3', 'sent 4'],
                                    {'sent 1': 1, 'sent 2': 0,
                                     'sent 3': 0, 'sent 4': 1},
                                    4)
    assert precision_at_4 == 0.5


def test_precision_at_k_for_no_annotated_matches():
    assert precision_at_k([],
                          {'sent 1': 1, 'sent 2': 0,
                           'sent 3': 0, 'sent 4': 1}) is None


def test_average_precision_for_single_query():
    avg_precision = average_precision(['sent 1', 'sent 2', 'sent 3', 'sent 4'],
                                      {'sent 1': 1, 'sent 2': 0,
                                       'sent 3': 0, 'sent 4': 1})
    assert avg_precision == 0.75


def test_average_precision_for_no_annotated_matches():
    assert average_precision([],
                             {'sent 1': 1, 'sent 2': 0,
                              'sent 3': 0, 'sent 4': 1}) is None


def test_map():
    annotations = {'prompt1': {('left', 'target1', 'right'): 1,
                               ('left', 'target2', 'right'): 0},
                   'prompt2': {('left', 'target3', 'right'): 1,
                               ('left', 'target4', 'right'): 1}}
    ranking = BeautifulSoup(
        """<section><h2>prompt1</h2>
<li>
<p>left<b>target2</b>right</p>
<p>left<b>target1</b>right</p></li></section>
<section><h2>prompt2</h2>
<li>
<p>left<b>target4</b>right</p>
<p>left<b>target3</b>right</p></li></section>""",
                   features='lxml').find_all('section')

    map_, df = mean_average_precision(['prompt1', 'prompt2'],
                                  annotations_by_prompt=annotations,
                                  ranking=ranking)
    assert map_ == 0.75, map_


def test_map_without_matches():
    annotations = {'prompt1': {('left', 'target1', 'right'): 1,
                               ('left', 'target2', 'right'): 0},
                   'prompt2': {('left', 'target3', 'right'): 1,
                               ('left', 'target4', 'right'): 0,
                               ('left', 'target5', 'right'): 1}}

    # with pytest.raises(Exception):
    map_, df = mean_average_precision(['prompt1', 'prompt2'],
                                      annotations_by_prompt=annotations)
    assert map_ is None and df['avg precision'].item() is None


def test_prepare_annotations():
    dict_ = {'prompt1': {'a': {'match': ['left', 'target1', 'right'],
                               'label': 1},
                         'b': {'match': ['left', 'target2', 'right'],
                               'label': 0}},
             'prompt2': {'a': {'match': ['left', 'target3', 'right'],
                               'label': 1},
                         'b': {'match': ['left', 'target4', 'right'],
                               'label': 0},
                         'c': {'match': ['left', 'target5', 'right'],
                               'label': 1}}}
    expected_out = {'prompt1': {('left', 'target1', 'right'): 1,
                                ('left', 'target2', 'right'): 0},
                    'prompt2': {('left', 'target3', 'right'): 1,
                                ('left', 'target4', 'right'): 0,
                                ('left', 'target5', 'right'): 1}}

    assert prepare_annotations(dict_) == expected_out
