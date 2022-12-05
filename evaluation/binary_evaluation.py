from typing import List, Dict
import bs4
import pandas as pd
import json
import re


def separate_query_and_context(html, label='relevant', return_label=False):
    sent = html.b.get_text(' ', strip=True).strip()
    left_context = html.get_text(' ', strip=True).strip()
    right_context = left_context
    sent_start = left_context.index(sent)
    left_context = left_context[:sent_start]
    right_context = right_context[sent_start + len(sent):].strip()
    if return_label:
        return {'match': (left_context, sent, right_context),
                'label': 1 if label == 'relevant' else 0}
    return (left_context.strip(),
            sent.strip(),
            right_context.strip())


def read_html(file):
    with open(file) as ifile:
        soup = bs4.BeautifulSoup(ifile.read(),
                                 features='lxml')
    return soup('section')


def read_json(file):
    with open(file) as ifile:
        prompt_rankings = json.load(ifile)
    return prompt_rankings


def index_sentences(sections):
    unique_sents = {}
    sents2ids = {}
    for section in sections:
        query = section.h2.get_text(' ', strip=True)
        query = ': '.join(query.split(': ')[1:])
        unique_sents[query] = []
        for li in section('li'):
            if li.p not in unique_sents[query]:
                unique_sents[query].append(li.p)
            if li('p')[-1] not in unique_sents[query]:
                unique_sents[query].append(li('p')[-1])
        sents2ids[query] = {separate_query_and_context(sent): i
                            for i, sent in
                            enumerate(unique_sents[query])}
        unique_sents[query] = {i: separate_query_and_context(sent)
                               for i, sent in
                               enumerate(unique_sents[query])}
    return unique_sents, sents2ids


def read_annotation_table(file):
    return pd.read_excel(file, engine='openpyxl')


table = read_annotation_table(
    '/Users/luisedu/Documents/Projekt/Causality/' +
    'causality_code/final_relevance_annotations.xlsx')
table.columns = [': '.join(col.split(': ')[1:]) for col in table.columns]
text = read_html(
    '/Users/luisedu/Documents/Projekt/Causality/causality_code/evaluation_text.html')
dummy_ranking = None#read_html('/Users/luidu652/Documents/causality_extraction/' +
                #          'dummy_ranking.html')

ids2sents, sents2ids = index_sentences(text)


def join_text_and_labels(text: List[bs4.element.Tag],
                         labels: pd.core.frame.DataFrame):
    """
    match labels to the corresponding sentence pairs
    :return: dictionary of joined labels and text by query
    """

    pairs = {}
    prompt2id = {}
    for col in labels.columns:
        if col == '0':
            continue
        # new columns
        if col.startswith('Prompt '):
            prompt = ': '.join(col.split(': ')[1:])
        else:
            prompt = col
            prompt2id[prompt] = col
            # print(col)
        for section in text:
            if re.match(rf'Prompt \d+b?: {col}$',
                        section.h2.get_text(' ', strip=True)):
                sent_ids = []
                for pair in section('li'):
                    sent_1 = separate_query_and_context(pair.p)
                    sent_2 = separate_query_and_context(pair('p')[-1])
                    sent_ids.append((sents2ids[col][sent_1],
                                     sents2ids[col][sent_2]))
                pairs[col] = list(zip(labels[col], sent_ids))
                break
    return pairs, prompt2id


def retrieve_relevance_judgements_per_sentence(
        text_label_pairs: Dict[str, List]):
    """
    extract binary judgement per sentence
    """
    binary_query_annotation = {}
    for query in text_label_pairs:

        current_query = {'relevant': [],
                         'irrelevant': []}
        for label, pair in text_label_pairs[query]:
            if label == 0:
                current_query['irrelevant'].append(pair[0])
                current_query['irrelevant'].append(pair[1])
            elif label == 1:
                current_query['relevant'].append(pair[0])
                current_query['irrelevant'].append(pair[1])
            elif label == 2:
                current_query['irrelevant'].append(pair[0])
                current_query['relevant'].append(pair[1])
            elif label in [3, 4, 5]:
                current_query['relevant'].append(pair[0])
                current_query['relevant'].append(pair[1])

        if not is_consistent(current_query, text_label_pairs[query]):
            #print(f'Inconsistent annotations for query {query}!')
            raise Exception(f'Inconsistent annotations for query {query}!')
        binary_query_annotation[query] = current_query
    return binary_query_annotation


def is_consistent(binary_judgements, pairs):
    consistent = True
    for sent in binary_judgements['relevant']:
        if sent in binary_judgements['irrelevant']:
            print(f'{sent} both judged relevant and irrelevant!')
            find_example(sent, pairs)
            consistent = False
    return consistent


def find_example(sent_id, pairs):
    for i, p in enumerate(pairs):
        if sent_id in p[1]:
            print(i, p)


def save_binary_annotations(binary_judgements,
                            filename='binary_judgments.json'):
    binary_judgements_sent_id = {}
    for query in binary_judgements:
        binary_judgements_sent_id[query] = {}
        query_key = query
        for id in binary_judgements[query]['relevant']:
            if not isinstance(ids2sents[query_key][id], tuple):
                binary_judgements_sent_id[query][id] = separate_query_and_context(
                    ids2sents[query_key][id], True)
            else:
                binary_judgements_sent_id[query][id] = {
                    'match': ids2sents[query_key][id], 'label': 1}
        for id in binary_judgements[query]['irrelevant']:
            if not isinstance(ids2sents[query_key][id], tuple):
                binary_judgements_sent_id[query][id] = separate_query_and_context(
                    ids2sents[query_key][id], True, 'irrelevant')
            else:
                binary_judgements_sent_id[query][id] = {
                    'match': ids2sents[query_key][id], 'label': 0}

    with open(filename, 'w') as ofile:
        json.dump(binary_judgements_sent_id, ofile)


def prepare_annotations(data='binary_judgments.json'):
    if isinstance(data, dict):
        judgements = data
    else:
        with open(data) as ifile:
            judgements = json.load(ifile)

    annotations_by_prompt = {}
    for prompt in judgements:
        annotations_by_prompt[prompt] = {}
        for id in judgements[prompt]:
            current = judgements[prompt][id]
            annotations_by_prompt[prompt][
                tuple([el.strip() for el in
                       current['match']])] = current['label']
    return annotations_by_prompt


def _binary_relevance_in_section(section, query, annotations):
    annotated_in_ranking = []
    relevant_in_ranking = []
    for i, match in enumerate(section):
        if not isinstance(match, tuple):
            match = separate_query_and_context(match)
        if match in annotations[query]:
            annotated_in_ranking.append(match)
            if annotations[query][match]:
                relevant_in_ranking.append(match)
    stats = {'annotated in ranking': len(annotated_in_ranking),
             'relevant in ranking': len(relevant_in_ranking),
             'N annotated': len(annotations[query]),
             'N relevant': sum(annotations[query].values())}
    return annotated_in_ranking, stats


def evaluate_binary_relevance(ranking, query, annotations):
    """
    take a ranking list and retain only annotated matches
    """

    for section in ranking:
        if isinstance(ranking, dict) and re.match(rf'(Prompt \d+b?: )?{query}$',
                                                  section):
            section = [tuple([s.strip() for s in el[1]]) for el in ranking[section]]
            return _binary_relevance_in_section(section, query, annotations)
        elif (isinstance(ranking, bs4.element.ResultSet) and
              re.match(rf'(Prompt \d+b?: )?{query}$',
                       section.h2.get_text(' ', strip=True))):
            section = section('p')
            return _binary_relevance_in_section(section, query, annotations)


def precision_at_k(system_ranking, annotations, k=10):
    ranking_size = len(system_ranking)
    n_relevant = sum(annotations.values())
    if k > ranking_size:
        print(f'ranking only covers {ranking_size} annotated sentences')
        if k > n_relevant:
            return None
    if ranking_size > 0:
        return sum([annotations[sent] for sent
                    in system_ranking[:k]]) / k


def average_precision(system_ranking: List, annotations: Dict) -> int:
    if system_ranking:
        summed_precision = 0
        for k in range(len(system_ranking)):
            summed_precision += (precision_at_k(system_ranking,
                                                annotations, k+1)
                                 * annotations[system_ranking[k]])
        num_relevant = sum(annotations.values())
        if num_relevant > 0:
            return summed_precision/num_relevant


def mean_average_precision(prompts, data='binary_judgments.json',
                           verbose=False,
                           annotations_by_prompt=None,
                           ranking=dummy_ranking):
    if annotations_by_prompt is None:
        annotations_by_prompt = prepare_annotations(data)
    avg_p = []
    indices = []
    overall_stats = {}
    for prompt in prompts:

        binary_relevance = evaluate_binary_relevance(
            ranking, prompt,
            annotations_by_prompt)
        if binary_relevance:
            annotated_ranks, stats = binary_relevance
        else:
            continue
        for k in stats:
            if k not in overall_stats:
                overall_stats[k] = []
            overall_stats[k].append(stats[k])
        p = average_precision(annotated_ranks,
                              annotations_by_prompt[prompt])
        if p is not None:
            avg_p.append(p)
            indices.append(prompt)
    df = pd.DataFrame(data={'avg precision': avg_p},
                      index=indices)
    if verbose:
        print(df)
    return df


def compute_precisions_at_k(prompts, data='binary_judgments.json',
                            verbose=False,
                            annotations_by_prompt=None,
                            ranking=dummy_ranking,
                            k=10,
                            iteration=0):
    """
    compute map for all given prompts.
    Includes a coverage and relevancy report on iteration 0
    """
    if annotations_by_prompt is None:
        annotations_by_prompt = prepare_annotations(data)
    p_at_k = []
    indices = []
    seen_prompts = []
    overall_stats = {}
    for prompt in prompts:
        binary_relevance = evaluate_binary_relevance(
            ranking, prompt,
            annotations_by_prompt)
        if binary_relevance:
            annotated_ranks, stats = binary_relevance
            seen_prompts.append(prompt)
        else:
            continue
        for key in stats:
            if key not in overall_stats:
                overall_stats[key] = []
            overall_stats[key].append(stats[key])

        p = precision_at_k(annotated_ranks,
                           annotations_by_prompt[prompt],
                           k=k)
        if p is not None:
            p_at_k.append(p)
            indices.append(prompt)
    # if len(p_at_k) > 0:
    #     mean_precision = sum(p_at_k)/len(p_at_k)
    # else:
    #     mean_precision = None
    for key in overall_stats:
        overall_stats[key].append(sum(overall_stats[key])
                                  / len(overall_stats[key]))
    df = pd.DataFrame(data={
        f'precision at {k}': p_at_k},
                      index=indices)
    if verbose:
        print(df)
    if iteration == 0:  # what is this about?
        stats = pd.DataFrame(data=overall_stats,
                             index=list(seen_prompts) + ['mean'])
        df = pd.merge(stats,
                      df, left_index=True, right_index=True)

    return df


if __name__ == '__main__':
    cols = list(table.columns)
    pairs_, prompts2ids = join_text_and_labels(text, table)  # table[cols[:34]])
    # pairs_ = join_text_and_labels(text, table[cols[:34]])
    pairs = retrieve_relevance_judgements_per_sentence(pairs_)
    save_binary_annotations(pairs, prompts2ids)
    mean_precision, ps_at_k = compute_precisions_at_k(cols[1:], verbose=True)
    # mean_precision, ps_at_k = compute_precisions_at_k(cols[1:34], verbose=True)
    # map_, df = mean_average_precision(cols[1:34], verbose=True)
    # df = pd.concat([ps_at_k, df], axis=1)
    # tester = {k:v for k, v in sents2ids.items() if 'Orsak' not in k and 'ö' not in k}
    # tester_ranking = read_json('13_query_ranking_model_0.json')
    # tester_stats = compute_precisions_at_k(tester.keys(), verbose=True, ranking=tester_ranking)
