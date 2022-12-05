"""
This script exemplifies a framework to evaluate relative ranking
in the context of causality-focused sentence retrieval.
"""
import pickle
from random import Random
from typing import List, Dict

import bs4
import pandas as pd
from binary_evaluation import separate_query_and_context,\
    join_text_and_labels#, sents2ids

# example annotations
toy_data = list(range(10))
eval_labels = [1, 2, 3, 1, 1, 1, 3]
sample_size = len(eval_labels)
# random number generator
random1 = Random(42)
random2 = Random(64)
first_sentences = random1.choices(toy_data, k=sample_size)
second_sentences = random2.choices(toy_data, k=sample_size)
eval_sent_ids = list(zip(first_sentences, second_sentences))

"""
Now it would probably make sense to check for consistency:
by the nature of the annotation task, there should not be duplicate
sentence pairs (per prompt).
Therefore, we don't check for directly conflicting annotations,
but for transitive ones - e.g. for the following example

sentence pairs: (a, b), (b,c), (a,c)
annotation: [1, 1, 2]

or to paraphrase:
 1) sentence a is more relevant than b
 2) sentence b is more relevant than c
 3) sentence c is more relevant than a

This means the last annotation introduces an inconsistency:
 if a is more relevant than b, which in turn is more relevant than c,
 then c should be less relevant than a.
"""


class RankingGraph:
    """
    A directed acyclic graph representation of the partial ranking
    that allows for efficient insertion and consistency checks.
    """
    def __init__(self):
        self.vertices = set()
        self.edges = set()
        self.edges_by_outnode = dict()
        self.edges_by_innode = dict()
        self.irrelevant = set()
        self.relevant = set()
        self.ranking = set()
        self.same_rank = dict()
        self.rank = None
        self.continuous_consistency_check = False

    def insert(self, pair, label):
        # add unseen ids to the set of vertices
        self.vertices = self.vertices.union(pair)
        edge = None
        # add the directed edge to the set of edges
        if label == 0:
            self.irrelevant = self.irrelevant.union(pair)
            if self.continuous_consistency_check:
                if not self.is_consistent():
                    raise Exception(
                        'insertion of irrelevant vertices renders ' +
                        'ranking inconsistent')

            return
        elif label in [1, 4]:
            edge = pair
            self.relevant.add(edge[0])
            if label == 1:
                self.irrelevant.add(edge[1])
            else:
                self.relevant.add(edge[1])
        elif label in [2, 5]:
            edge = tuple([pair[1], pair[0]])
            self.relevant.add(edge[0])
            if label == 2:
                self.irrelevant.add(edge[1])
            else:
                self.relevant.add(edge[1])
        elif label == 3:
            self.insert_at_same_rank(pair)
            self.relevant.add(pair[0])
            self.relevant.add(pair[1])
            # combine the ranking of both nodes if previously seen
            # and in the future
        if edge:
            self.add_new_edge(edge)
        self.update_edges()

    def insert_at_same_rank(self, edge):
        if edge[0] not in self.same_rank:
            self.same_rank[edge[0]] = set()
        if edge[1] not in self.same_rank:
            self.same_rank[edge[1]] = set()
        self.same_rank[edge[0]].add(edge[1])
        self.same_rank[edge[1]].add(edge[0])

    def update_edges(self):
        # combine the ranking of equally relevant nodes if previously seen
        # and in the future
        # update all equally relevant nodes first
        for node in self.same_rank:
            for similar_node in self.same_rank[node]:
                if similar_node in self.same_rank:
                    self.same_rank[node] = self.same_rank[node].union(
                        self.same_rank[similar_node])
                    self.same_rank[node] = self.same_rank[node].difference(
                        [node])
        for node in self.same_rank:
            for similar_node in self.same_rank[node]:
                if similar_node in self.edges_by_outnode:
                    # copy relation to lower ranked nodes
                    for innode in self.edges_by_outnode[similar_node]:
                        self.add_new_edge(tuple([node, innode]))
                if similar_node in self.edges_by_innode:
                    # copy relation to higher ranked nodes
                    for outnode in self.edges_by_innode[similar_node]:
                        self.add_new_edge(tuple([outnode, node]))

    def add_new_edge(self, edge):
        # check if contradicting edge exists
        outnode, innode = edge
        if outnode in self.irrelevant:
            raise Exception(
                'conflicting annotation: tried to add edge from previously ' +
                f'irrelevant node {outnode}')
        inverted_tuple = tuple([innode, outnode])
        if innode in self.edges_by_outnode\
           and outnode in self.edges_by_outnode[innode]:
            raise Exception(
                f'conflicting edge found: tried to add {edge} ' +
                f'and found existing {inverted_tuple}')

        # if outnode in self.edges_by_outnode\
        #    and innode in self.edges_by_outnode[outnode]:
        #     raise Exception(f'duplicate edge: {edge} already in graph')
        else:
            self.edges.add(edge)
            if outnode not in self.edges_by_outnode:
                self.edges_by_outnode[outnode] = set()
            self.edges_by_outnode[outnode].add(innode)
            if innode not in self.edges_by_innode:
                self.edges_by_innode[innode] = set()
            self.edges_by_innode[innode].add(outnode)
        if self.continuous_consistency_check:
            if not self.is_consistent():
                raise Exception(f'new edge {edge} introduced inconsistency!')

    def is_consistent(self):
        # make sure there are no cycles
        for outnode in self.edges_by_outnode:
            if self._has_cycle(outnode, outnode):
                return False
        for vertex in self.irrelevant:
            if vertex in self.edges_by_outnode:
                return False
        return True

    def _has_cycle(self, innode, outnode):
        """
        recursively check that a node does not connect to itsself
        """
        if innode in self.edges_by_outnode:
            if outnode in self.edges_by_outnode[innode]:
                return True
            else:
                for in_ in self.edges_by_outnode[innode]:
                    if self._has_cycle(in_, outnode):
                        return True
        return False

    def hierarchies(self):
        """
        build partial rankings based on current graph
        """
        ranking = {}
        seen = set()
        # identify chains of connected vertices
        for node in self.edges_by_outnode:
            if node not in seen:
                ranking[node] = self.extract_node_rankings(node)
                visited_nodes = set([node_ for list_ in ranking[node]
                                     for node_ in list_])
                seen = seen.union(visited_nodes)
        rankings = [''.join(partial_ranking) for node_ranking
                    in ranking.values()
                    for partial_ranking in node_ranking]
        return remove_sublists(rankings)

    def extract_node_rankings(self, node):
        rankings = []
        if node in self.edges_by_outnode:
            for innode in self.edges_by_outnode[node]:
                partial_ranking = self.extract_node_rankings(innode)
                rankings.extend([[str(node)] + partial for partial in
                                 partial_ranking])
            return rankings
        else:
            return [[str(node)]]

    def count_agreements(self, ranking: List[int],
                         explicit_only: bool = False) -> int:
        """
        count number of ranked pairs in ranking that are
        in agreement with the graph.
        :param ranking: a ranked list of sentence indeces
        :param explicit_only: mode to consider only explicitly annotated pairs,
        also ranks inferred implicit rankings if False
        :return: number of similarly ranked pairs
        """

        if explicit_only:
            return self._rank_explicit_pairs(ranking)
        else:
            return self._rank_implicit_pairs(ranking)
        return 0

    def _rank_explicit_pairs(self, ranking, return_precision=False):
        """
        currently this still ranks inferred pairs
        from similarly relevant pairs
        """
        agreements = 0
        seen = set()
        for relevant, less_relevant in self.edges:
            agreements += self._rank_pair(relevant, less_relevant,
                                          ranking, seen)
        if return_precision:
            return agreements / len(seen)
        return agreements

    def _rank_implicit_pairs(self, ranking, return_precision=False):
        agreements = 0
        gold_rankings = self.hierarchies()
        seen = set()
        for partial_ranking in gold_rankings:
            for i, relevant in enumerate(partial_ranking):
                for less_relevant in partial_ranking[i+1:]:
                    agreements += self._rank_pair(relevant, less_relevant,
                                                  ranking, seen)
                for irrelevant in self.irrelevant:
                    agreements += self._rank_pair(relevant, irrelevant,
                                                  ranking, seen)
        if return_precision:
            return agreements / len(seen)
        return agreements

    def _rank_pair(self, relevant, less_relevant, ranking, seen):
        if (relevant, less_relevant) not in seen:
            assert relevant in ranking and less_relevant in ranking,\
                'Ranking is missing one or more annotated sentences ' +\
                f'{relevant}, {less_relevant}'
            seen.add((relevant, less_relevant))
            if ranking.index(relevant) < ranking.index(less_relevant):
                # print(relevant, less_relevant)
                return 1
        return 0

    def precision(self, ranking, explicit_only=False):
        if explicit_only:
            return self._rank_explicit_pairs(ranking, True)
        return self._rank_implicit_pairs(ranking, True)

    def assign_rank(self):
        """
        set rank values for all nodes following EDRC
        """
        self.rank = {vertex: 0 for vertex in self.vertices}
        self.discount = self.rank
        for node in self.vertices:
            if node not in self.edges_by_innode:
                self._visit(node, 0)

    def _visit(self, node, rank):
        """
        assign rank to node and all of the nodes reached by its
        outcoming edges
        """
        self.rank[node] = max(self.rank[node], rank + 1)
        if node in self.edges_by_outnode:
            for innode in self.edges_by_outnode[node]:
                self._visit(innode, self.rank[node])

    def infer_ranking(self):
        """
        infer relevance ranking for unseen combinations of
        confirmed relevant and irrelevant sentences
        """
        for first_node in self.relevant:
            assert first_node not in self.irrelevant
            for second_node in self.irrelevant:
                edge = (first_node, second_node)
                if edge not in self.edges:
                    self.add_new_edge(edge)


def remove_sublists(rankings):
    new_rankings = []
    for ranking in rankings:
        is_in_other_ranking = False
        for other_ranking in rankings:
            if ranking is not other_ranking\
               and ranking in other_ranking:
                is_in_other_ranking = True
                break
        if not is_in_other_ranking:
            new_rankings.append(tuple([int(node) for node in ranking]))
    return new_rankings


# def is_inconsistent(partial_ranking):
#     graph = RankingGraph()
#     graph.insert((2, 4), 2)
#     graph.insert((1, 4), 3)
#     graph.insert((2, 3), 1)
#     graph.insert((5, 4), 1)
#     graph.insert((0, 4), 2)
#     graph.insert((0, 6), 0)
#     graph.insert((5, 6), 1)

def convert_ranking_to_graph(ranking: List,
                             sents2ids: Dict[str, int]) -> RankingGraph:
    """
    create a RankingGraph for the system ranking of a single prompt
    """
    graph = RankingGraph()
    for i, first_sent in enumerate(ranking):
        # check if sentence in sents2ids
        if not isinstance(first_sent, tuple):
            first_sent = separate_query_and_context(first_sent)
        if first_sent in sents2ids:
            for second_sent in ranking[i+1:]:
                if not isinstance(second_sent, tuple):
                    second_sent = separate_query_and_context(second_sent)
                if second_sent in sents2ids and second_sent != first_sent:
                    graph.insert((sents2ids[first_sent],
                                  sents2ids[second_sent]), 4)
    return graph


def convert_rankings_to_graph(ranking, sents2ids, prompts2ids):
    graphs_by_prompt = {}
    for prompt_ranking in ranking:
        if isinstance(prompt_ranking, str):
            prompt = prompt_ranking
            current_ranking = [tuple([el.strip() for el in text]) for score, text
                               in ranking[prompt]]

        else:
            prompt = prompt_ranking.h2.get_text(' ', strip=True)
            current_ranking = list(set(prompt_ranking('p')))
        prompt = prompt.lstrip('Prompt 1234567890b: ')
        # print(f'extracting: "{prompt}"')
        prompt_key = prompt
        graphs_by_prompt[prompt] = convert_ranking_to_graph(
            current_ranking,
            sents2ids[prompt_key])
    return graphs_by_prompt


def create_graphs_for_prompts(sentences, annotations=None):
    graphs_by_prompt = {}
    if isinstance(sentences, list):
        joined_data, prompts2ids = join_text_and_labels(sentences,
                                                        annotations)
    else:
        # alternatively convert json
        joined_data = sentences
        prompts2ids = None
    for prompt, annotations in joined_data.items():
        graph = RankingGraph()
        for label, pair in annotations:
            graph.insert(pair, label)
        graphs_by_prompt[prompt] = graph
    return graphs_by_prompt, prompts2ids


class EDRC:

    def __init__(self, gold_standard, predictions):
        self.gold = gold_standard
        if self.gold.rank is None:
            self.gold.assign_rank()
        self.pred = predictions

    def accuracy(self):
        """
        compute accuracy of relevance pairs
        (this is not needed for EDRC, but convenient to add here)
        """
        if len(self.gold.edges) > 0:
            return len([edge for edge in self.gold.edges
                        if edge in self.pred.edges]) / len(self.gold.edges)

    def edrc(self, verbose=False):
        S = set([node for node in self.gold.vertices
                 if node not in self.gold.edges_by_innode])
        innodes = self.gold.vertices.difference(S)
        # higher ranked nodes or nodes with unknown relation to
        # a given node for each node with incoming edge
        W = {node: [v for v in self.gold.vertices if
                    (node not in self.gold.edges_by_outnode or
                     v not in self.gold.edges_by_outnode[node])
                    and v != node]
             for node in innodes}
        # normalisation factor
        Z = sum([len(W[node])/self.gold.rank[node]
                 for node in innodes])
        if verbose:
            print('S:', S)
            print('gold/S:', innodes)
            print('z:', Z)
            print('w:', W)
        if Z != 0:
            rank_correlation = sum(
                [self.score(node, W[node], verbose)/self.gold.discount[node]
                 for node in innodes])
            if verbose:
                print('rank correlation:', rank_correlation)
                print(f'EDRC: {(2/Z) * rank_correlation - 1}')
            return (2/Z) * rank_correlation - 1

    def expected_score(self, node, higher_ranked):
        # edge is known
        edge = (higher_ranked, node)
        if edge in self.gold.edges:
            if edge in self.pred.edges:
                return 1
            elif (node, higher_ranked) in self.pred.edges:
                return 0
            else:
                return 0.5
        elif (node, higher_ranked) in self.gold.edges:
            print('this should not happen!', node, higher_ranked)
        # unknown preference
        else:
            # if edge in self.pred.edges:
            # return 1
            # I think it is always 0.5 if there is no prediction?
            return 0.5

    def score(self, vertex, W, verbose=False):
        # set of higher ranked items or items of unknown ranking
        ep = [self.expected_score(vertex, node) for node in W]
        if verbose:
            print(vertex, W)
            print(ep, '/', self.gold.discount[vertex])
        return sum(ep)


def edrc_per_prompt(ranking, text, sents2ids, labels=None, name=None):
    gold_graphs, prompts2ids = create_graphs_for_prompts(text, labels)
    predicted_graphs = convert_rankings_to_graph(ranking,
                                                 sents2ids, prompts2ids)
    edrc_dict = {}
    if name is not None:
        with open(f'{name}.gold.edrc.pickle','wb') as ofile:
            pickle.dump(gold_graphs, ofile)
        with open(f'{name}.pred.edrc.pickle', 'wb') as ofile:
            pickle.dump(predicted_graphs, ofile)
        with open(f'{name}.sents2ids.pickle', 'wb') as ofile:
            pickle.dump(sents2ids, ofile)
    for prompt in predicted_graphs:
        print(prompt)
        if prompt in gold_graphs:
            edrc = EDRC(gold_graphs[prompt], predicted_graphs[prompt])
            edrc_dict[prompt] = [edrc.edrc(), EDRC(gold_graphs[prompt],
                                                   gold_graphs[prompt]).edrc()]
            edrc_dict[prompt].append(edrc.accuracy())
            
            # inferring implicit rankings
            gold_graphs[prompt].infer_ranking()
            edrc = EDRC(gold_graphs[prompt], predicted_graphs[prompt])
            edrc_dict[prompt].extend([edrc.edrc(), EDRC(gold_graphs[prompt],
                                     gold_graphs[prompt]).edrc()])
            edrc_dict[prompt].append(edrc.accuracy())
        else:
            print(f'prompt {prompt} not found in annotations')
    df = pd.DataFrame(data=edrc_dict)
    df = df.T
    df.columns = ['explicit only', '(explicit identical graphs)', 'accuracy explicit',
                  'implicit pairs', '(implicit identical graphs)', 'accuracy implicit']
    # df.to_excel('edrc_test.xlsx', engine='openpyxl')
    return df

if __name__ == '__main__':

    gold = RankingGraph()
    for edge in [('a', 'b'), ('a', 'd')]:
        gold.insert(edge, 1)
    gold.assign_rank()
    pred = RankingGraph()
    for edge in [('a', 'b'), ('a', 'd')]:
        pred.insert(edge, 1)


    edrc = EDRC(gold, pred)
    result = edrc.edrc(True)
    edrc = EDRC(pred, gold)
    result = edrc.edrc(True)

    # significance testing
    from scipy.stats import ttest_rel
    # ttest_rel(list(custom_ct['accuracy.1'][:-1]), list(ct['accuracy.1'][:-4]), nan_policy='omit')
    file_specs = ['random', 'Rise_ct', 'custom', 'STSb2', 'STSb1_sou_ct', 'STSb2_sou_ct']

    # score = 'accuracy.1'
    # score = 'avg precision'
    # score = 'implicit pairs'
    score = 'precision at 10'
    print(score)
    print('(model1, model2)\tstatistic\tpvalue')
    for i, specifier_1 in enumerate(file_specs):
        m1 = pd.read_excel(f'evaluation/results/{specifier_1}_eval_table.xlsx',
                           engine='openpyxl')
        for specifier_2 in file_specs[i+1:]:
            m2 = pd.read_excel(f'evaluation/results/{specifier_2}_eval_table.xlsx',
                               engine='openpyxl')
            res = ttest_rel(list(m1[score][:-1]), list(m2[score][:-1]), nan_policy='omit')
            print(f'{specifier_1, specifier_2}: {res.statistic}  {res.pvalue}')
