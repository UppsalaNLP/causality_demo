from typing import List

import json
import tqdm
import bs4
import pandas as pd
import sys

sys.path.append('/Users/luidu652/Documents/evaluation')
from binary_evaluation import prepare_annotations, join_text_and_labels,\
    retrieve_relevance_judgements_per_sentence, compute_precisions_at_k,\
    mean_average_precision, dummy_ranking, save_binary_annotations,\
    read_annotation_table, read_html, read_json, index_sentences
from evaluation import edrc_per_prompt

path = '/Users/luisedu/Documents/Projekt/Causality/causality_code/evaluation'
# table = read_annotation_table(
#     '/Users/luidu652/Documents/causality_extraction/' +
# #    'causality_demo/pilot_annotation/agreement_annotation_table.xlsx')
#     'causality_demo/pilot_annotation/changed_agreement_annotation_table.xlsx')
# table.columns = [': '.join(col.split(': ')[1:]) for col in table.columns]
# text = read_html(
#     '/Users/luidu652/Documents/causality_extraction/evaluation_text.html')
# ids2sents, sents2ids = index_sentences(text)

def load_dataset(path):
    '''
    load dataset from json and convert to paired and individual relevance
    '''
    with open(path) as ifile:
        dataset = json.load(ifile)
    pairs = {}
    sents2ids = {}
    n_sents = 0
    for id, pair_entry in dataset.items():
        prompt = pair_entry['prompt']
        if prompt not in pairs:
            pairs[prompt] = []
        s1 = (pair_entry['sentence_1']['left context'],
              pair_entry['sentence_1']['target sentence'],
              pair_entry['sentence_1']['right context'])
        s2 = (pair_entry['sentence_2']['left context'],
              pair_entry['sentence_2']['target sentence'],
              pair_entry['sentence_2']['right context'])
        if prompt not in sents2ids:
            sents2ids[prompt] = {}
        if s1 not in sents2ids[prompt]:
            sents2ids[prompt][s1] = len(sents2ids[prompt])
        if s2 not in sents2ids[prompt]:
            sents2ids[prompt][s2] = len(sents2ids[prompt])
        pairs[prompt].append((pair_entry['annotation'], (sents2ids[prompt][s1], sents2ids[prompt][s2])))
        n_sents += 2
    individual_sentences = retrieve_relevance_judgements_per_sentence(pairs)
    return pairs, individual_sentences, sents2ids


def evaluation_report(#text: List[bs4.element.Tag],
                      #labels: pd.core.frame.DataFrame,
        data_path: str,
                      k: List[int], map: bool = True,
                      edrc=True,
                      ranking: bs4.element.ResultSet = dummy_ranking,
                      verbose: bool = False,
                      #data: str = '/Users/luidu652/Documents/evaluation/binary_judgments.json',
                      outfile: str = None,
                      edrc_name=None
                      ) -> pd.core.frame.DataFrame:

    # TODO eliminate need for files beyond the json
    # needed, binary judgement per sent
    # 6 cat label per pair/prompt -> list of prompts
    
    # annotations_by_prompt = prepare_annotations(data)
    # prompts = list(labels.columns)
    # prompts.remove('0')
    # pairs, prompts2ids = join_text_and_labels(text, labels)
    # pairs = retrieve_relevance_judgements_per_sentence(pairs)
    # save_binary_annotations(pairs, prompts2ids)
    pairs, binary_judgements, sents2ids = load_dataset(data_path)
    prompts = pairs.keys()
    table = pd.DataFrame()
    for i, n in enumerate(k):
        ps_at_k = compute_precisions_at_k(
            prompts,
            #binary_judgements,
            verbose=verbose,
            k=n,
            #annotations_by_prompt=annotations_by_prompt,
            ranking=ranking,
            iteration=i)
        table = pd.concat([table, ps_at_k], axis=1)
    if map:
        df = mean_average_precision(
            prompts, verbose=verbose,
            ranking=ranking)
            #annotations_by_prompt=annotations_by_prompt)
        table = pd.concat([table, df], axis=1)
    if edrc:
        edrc_table = edrc_per_prompt(ranking, pairs, sents2ids,#labels,
                                     name=edrc_name)
        table = pd.concat([table, edrc_table], axis=1)
    # fix table
    # ignore irrelevant prompt
    relevant_prompts = table[table['relevant in ranking'] > 0]
    mean = {}
    for column in table.columns:
        if column.startswith('precision at'):
            n = int(column.split()[-1])
            mean[column] = relevant_prompts[relevant_prompts['relevant in ranking'] >= n][column].mean()
        else:
            mean[column] = relevant_prompts[column].mean()
        # if column == 'accuracy':
        #     break
    table.loc['mean'] = mean
    if outfile:
        table.to_excel(outfile, engine='openpyxl')
    return table


def compute_percentages_on_full_ranking(text, labels, model, embeddings):
    """
    generate full ranking per prompt and report coverage percentage
    of relevant/annotated? matches for top 100, 500, 1000
    """
    annotations_by_prompt = prepare_annotations(data)
    prompts = list(labels.columns)
    prompts.remove('0')
    pairs, prompts2ids = join_text_and_labels(text, labels)
    pairs = retrieve_relevance_judgements_per_sentence(pairs)
    save_binary_annotations(pairs, prompts2ids)
    table = pd.DataFrame()
    ids2sents, sents2ids = index_sentences(text)


def assess_full_ranking(in_data='Rise_ct_rank_percentages.json'):
    table = pd.read_json(f'/Users/luidu652/Documents/evaluation/{in_data}')
    flipped_data = {'relevant': {}, 'irrelevant': {}, 'annotated':{}}
    for k in flipped_data:
        for p in table:
            flipped_data[k][p] = table[p][k]

    relevant = pd.DataFrame(data=flipped_data['relevant'])
    irrelevant = pd.DataFrame(data=flipped_data['irrelevant'])
    annotated = pd.DataFrame(data=flipped_data['annotated'])
    t = {}
    for prompt in relevant:
        for index, value in relevant[prompt].items():
            if prompt not in t:
                t[prompt] = {}
            if max(relevant[prompt]) > 0:
                t[prompt][f'relevant at {index}'] = value / max(relevant[prompt])
            t[prompt][f'irrelevant at {index}'] = irrelevant[prompt][index] / max(irrelevant[prompt])
            t[prompt][f'annotated at {index}'] = annotated[prompt][index] / max(annotated[prompt])

    percentages = pd.DataFrame(data=t)
    percentages.T.to_excel(f'{in_data.split("_")[0]}rank_percentage.xlsx', engine='openpyxl')


def full_ranking_indeces(in_data='43_full_prompt_ranking_model_STSbertb1.json'):
    table = pd.read_json(f'/Users/luidu652/Documents/evaluation/{in_data}')
    flipped_data = {'relevant': {}, 'irrelevant': {}, 'annotated':{}}
    for k in flipped_data:
        for p in table:
            flipped_data[k][p] = table[p][k]

    relevant = pd.DataFrame(data=flipped_data['relevant'])
    irrelevant = pd.DataFrame(data=flipped_data['irrelevant'])
    annotated = pd.DataFrame(data=flipped_data['annotated'])
    t = {}
    for prompt in relevant:
        for index, value in relevant[prompt].items():
            if prompt not in t:
                t[prompt] = {}
            if max(relevant[prompt]) > 0:
                t[prompt][index] = value / max(relevant[prompt])
            t[prompt][index] = irrelevant[prompt][index] / max(irrelevant[prompt])
            t[prompt][index] = annotated[prompt][index] / max(annotated[prompt])

    percentages = pd.DataFrame(data=t)
    percentages.T.to_excel(f'{in_data.split("_")[0]}rank_percentage.xlsx', engine='openpyxl')


def precision_at_k(table, k):
    if k not in [5, 10, 20, 40]:
        raise Exception(f'precision at {k} has not been pre-computed')
    prompts = table.loc[table['Unnamed: 0'] != 'mean']
    relevant_prompts = prompts.loc[prompts['N relevant'] >= k]
    if len(relevant_prompts) > 0:
        print(f'prompts with at least {k} relevant examples: {len(relevant_prompts)}')
        precision = sum(relevant_prompts[f'precision at {k}']) / len(relevant_prompts)
        print(f'precision at {k}: {precision}')


if __name__ == '__main__':
    # e.g.

    # load gold data
    data = load_dataset('/Users/luisedu/Documents/Projekt/Causality/Swedish-Causality-Datasets/Curated-Ranking-Data-Set/ranking_data_set.json')
    data_path = '/Users/luisedu/Documents/Projekt/Causality/Swedish-Causality-Datasets/Curated-Ranking-Data-Set/ranking_data_set.json'
    path = '/Users/luisedu/Documents/Projekt/Causality/causality_code/evaluation'

    save_out = False
    outname = None
    avg_scores = {}
    for i in range(1, 11):
        ranking_file = f'{path}/{i}_43_query_ranking_model_random.json'
        if save_out:
            outname = f'{path}/results2022/random_{i}_eval_table.xlsx'
        df = evaluation_report(data_path, [5, 10], edrc=True,
                               ranking=read_json(ranking_file),
                               edrc_name=f'random_{i}',
                               outfile=outname)
        avg_scores[i] = df.T['mean']
    pd.DataFrame(avg_scores).mean(axis=1).to_excel(f'{path}/results2022/random_10_avg.xlsx')

    for out, ranking in [('ct22', 0),
                         ('STSbertb1', 'STSbertb1'),
                         ('STSbertb2', 'STSbertb2'),
                         ('STSbertb1_sou', 'STSbertb1_sou_tuning'),
                         ('STSbertb2_sou', 'STSbertb2_sou_tuning'),
                         ('BERT', 'KBbert-base-swedish-cased')
                         ]:
        evaluation_report(data_path, [5, 10], edrc=True,
                          ranking=read_json(f'{path}/43_query_ranking_model_{ranking}.json'),
                          edrc_name=None,
                          outfile=f'{path}/results2022/{out}_eval_table.xlsx')
    # test individual prompts
    prompts = [
        '"bero på"',
        '"bidra till"', 
        'framkalla', 
        'förorsaka', 
        '"leda till"', 
        'medföra', 
        'orsaka', 
        '"på grund av"',
        'påverka', 
        'resultera', 
        '"till följd av"',
        '"vara ett resultat av"', 
        'vålla',
    ]
    avg_scores = {}
    for prompt in prompts:
        ranking_file = f'{path}/{prompt}_43_query_ranking_model_Contrastive-TensionBERT-Base-Swe-CT-STSb.json'
        df = evaluation_report(data_path, [5, 10], edrc=True,
                               ranking=read_json(ranking_file),
                               edrc_name=f'{prompt}',
                               outfile=None)
        avg_scores[prompt] = df.T['mean']
        pd.DataFrame(avg_scores).T.to_excel(f'{path}/results2022/individual_prompts_avg.xlsx')
