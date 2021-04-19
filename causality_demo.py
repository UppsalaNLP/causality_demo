import csv
import base64
import time
from tqdm import tqdm
import sys
from io import BytesIO
import pickle
from collections import OrderedDict
import gzip
import os
import re
import math
import logging
from urllib import parse

import streamlit as st
from streamlit.hashing import _CodeHasher
import pandas as pd
import torch

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server
# import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
# from memory_profiler import profile
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.DEBUG)
# logging.root.setLevel(logging.DEBUG)
logging.root.setLevel(logging.NOTSET)

st.set_page_config(page_title='demo app',
                   page_icon=':mag:',
                   layout='centered',
                   initial_sidebar_state='expanded')


def get_table_download_link(table, query):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    table.to_excel(writer, index=False, sheet_name='Sökresultat',
                   float_format="%.2f", startrow=2)
    worksheet = writer.sheets['Sökresultat']
    worksheet.write_string(0, 0, query)

    writer.save()
    output_val = output.getvalue()
    b64 = base64.b64encode(output_val)

    link = f'<a href="data:application/octet-stream;base64,{b64.decode()}" ' +\
        f'download="sökresultat.xlsx">spara {len(table)} resultat</a>'
    return link


@st.cache
def generate_prompts(cause=None, effect=None):
    """
    insert topic and expansions into prompt templates.
    This is a local version that does not include query expansion
    for efficiency reasons.
    """

    prompt_dict = {
        '"bero på"': ['X beror på [MASK]'],
        '"bidra till"': ['[MASK] bidrar till X'],
        'framkalla': ['[MASK] framkallar X'],
        'förorsaka': ['[MASK] förorsakar X'],
        '"leda till"': ['[MASK] leder till X'],
        'medföra': ['[MASK] medför X'],
        'orsaka': ['[MASK] orsakar X'],
        '"på grund av"': ['X på grund av [MASK]',
                          'X händer på grund av [MASK]'],
        'påverka': ['[MASK] påverkar X'],
        'resultera': ['[MASK] resulterar i X'],
        '"till följd av"': ['X till följd av [MASK]',
                            'X händer till följd av [MASK]'],
        '"vara ett resultat av"': ['X är ett resultat av [MASK]'],
        'vålla': ['[MASK] vållar X']}

    templates = [template
                 for keyword_templates in prompt_dict.values()
                 for template in keyword_templates]

    def fill_templates(term, templates, placeholder='X'):
        return [template.replace(placeholder, term) for template in templates]

    # We used query expansion before, if we ever go back, we need to
    # tokenize differently so that neighbours representing part of a word
    # e.g. ##klimat get treated correctly.
    # Alternatively we could filter them out. Maybe that is better since
    # using them unaltered would lead to weird combinations depending on
    # template and position:
    # (i.e. ['[MASK]', 'orsakar', '##klimat'] -> '[MASK] orsakarklimat'!?)
    # topic_terms = [topic]

    # generate prompts
    prompts = []
    if effect:

        prompts = [prompt for prompt in fill_templates(effect, templates)]
        if cause:
            prompts = [prompt for prompt in
                       fill_templates(cause, prompts, '[MASK]')]
    elif cause:
        prompts = [prompt for prompt in
                   fill_templates(cause, templates, '[MASK]')]
        prompts = [prompt for prompt in fill_templates('[MASK]', prompts)]

    return prompts


@st.cache(allow_output_mutation=True)
def load_binary(emb_file):
    logging.debug(f'loading {emb_file}')
    start = time.time()
    if emb_file.endswith('.gzip') or emb_file.endswith('.gz'):
        with gzip.GzipFile(emb_file, 'rb') as ifile:
            embeddings = pickle.loads(ifile.read())
    elif emb_file.endswith('.pickle'):
        with open(emb_file, 'rb') as ifile:
            embeddings = pickle.load(ifile)
    else:
        raise RuntimeError(f'unknown file type {emb_file}')
    logging.debug(f'load_binary() took {time.time()-start} s ')
    return embeddings


@st.cache(allow_output_mutation=True)
# @profile
def load_documents(input_emb, input_meta):
    """
    load prefiltered text and embeddings.
    """
    start = time.time()
    logging.debug('loading embeddings')
    docs = {}
    docs['embeddings'] = load_binary(input_emb)
    if input_meta.endswith('.gz'):
        docs['meta'] = load_binary(input_meta)
    else:
        with open(input_meta) as ifile:
            reader = csv.reader(ifile, delimiter=';')
            docs['meta'] = [line for line in reader]
    logging.debug(f'load_documents() took {time.time()-start} s ')
    nn = fit_nn_model(docs)
    return docs, nn


@st.cache(allow_output_mutation=True)
def init_ct_model(selected_model):
    from transformers import AutoModel, AutoTokenizer
    import torch

    logging.debug('loading model')
    on_gpu = torch.cuda.is_available()
    logging.debug(f'GPU available: {on_gpu}')
    model_name = tok_name = "Contrastive-Tension/BERT-Base-Swe-CT-STSb"
    if selected_model == 1:
        pass # use new model
    logging.debug(f'loading tokeniser: {tok_name}')
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    logging.debug(f'loading BERT model: {model_name}')
    model = AutoModel.from_pretrained(model_name, from_tf=True)
    return on_gpu, tokenizer, model


@st.cache
# @profile
def embed_text(samples, prefix='', save_out=True, selected_model=0):
    """
    embed samples using the swedish STS model
    """
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    on_gpu, tokenizer, model = init_ct_model(selected_model)
    model.eval()
    logging.debug(f'embedding {len(samples)} sentences ...')
    embeddings = []
    batch_size = 100
    with torch.no_grad():
        if on_gpu:
            model.to('cuda')
        dataloader = [samples]
        if len(samples) > batch_size:
            dataloader = DataLoader(
                samples,
                sampler=SequentialSampler(samples),
                batch_size=batch_size
            )
            logging.debug(f'{math.ceil(len(samples)/batch_size)} batches')
        for i, batch in enumerate(dataloader):
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512)
            if on_gpu:
                inputs = inputs.to('cuda')
            out = model(**inputs)
            b_embeddings = mean_pool(out, inputs['attention_mask'])
            embeddings.append(b_embeddings.cpu())
        embeddings = torch.cat(embeddings)
    if save_out:
        filename = f'{prefix}{len(samples)}_embeddings.gzip'
        with gzip.GzipFile(filename, 'wb') as embeddings_out:
            embeddings_out.write(pickle.dumps(embeddings))
            logging.debug(f'saved embeddings to {filename}')
        return embeddings, filename
    logging.debug('done')
    return embeddings


def mean_pool(model_out, input_mask):
    """following sentence_transformers"""
    import torch

    embeddings = model_out[0]
    attention_mask = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * attention_mask, 1)
    n = torch.clamp(attention_mask.sum(1), min=1e-9)
    return sum_embeddings / n


# @st.cache
def display_result(state, term, doc_id, filter, seen_documents, match):
    """
    display a single match if it matches the filter
    """
    # start = time.time()

    key = doc_id
    if isinstance(doc_id, tuple):
        text, doc_id, sent_nb, match_emb_id = doc_id
    doc_id = doc_id.split('_')[-1].split('.')[0]

    if doc_id in seen_documents:
        return False
    stats = ['rank', 'distance', 'nb_matches']
    doc_title, date = ids2doc[doc_id]
    # publishing date
    # date = datetime.datetime.fromisoformat(date)
    # SOU number-based date
    year = doc_title.split()[-1].split(':')[0]
    assert year.isnumeric(), f'malformed document title {doc_title} ({year})'
    year = int(year)
    # match = state.ranking[(term, state.scope, state.top_n_ranking,
    #                       ' '.join(state.rank_by))][key]

    def rank_func(x):
        filters = []
        for filter in state.rank_by:
            if filter == 'average rank':
                filters.append(-(sum(match['matched_text'][x]['rank']) /
                                 len(match['matched_text'][x]['rank'])))
            if filter == 'average distance':
                filters.append(1-(match['matched_text'][x]['distance'] /
                                  len(match['matched_text'][x]['distances'])))
        if not state.rank_by:
            filters = -1*match['matched_text'][x]['distance']

        return filters

    def format_stats(k, match=match):
        return (": ".join(
            [k, f"{match[k]:>1.3f}"])
                if isinstance(match[k], float)
                else ": ".join([k,
                                f"{match[k]}"#sum(match[k]) / len(match[k]):>1.3f}"
                                if isinstance(match[k], list)
                                else str(match[k])]))\
                                     if k in match else ''

    if year in range(filter['time_from'], filter['time_to'])\
       and (state.doc_id is None or state.doc_id != doc_id):
        # if we extract page ids from the html we might even be able to
        # link the approximate location of the match
        # (ids seem to be a little off)

        displayed_sents = 0
        doc_title_text = doc_title
        continuation = re.findall(r'd(\d+)$', doc_id)
        html_link = f'https://data.riksdagen.se/dokument/{doc_id}'
        if continuation:
            doc_title_text = f'{doc_title}, del {continuation[0]}'

        if 'matched_text' in match.keys():
            state.outpage.append(f'## [{doc_title_text}]({html_link})')
            stats_header = f'({", ".join([format_stats(k) for k in stats])})'
            state.outpage.append(f'### {stats_header}')
            for sent in sorted(match['matched_text'],
                               key=rank_func, reverse=True):
                sentence_match = match['matched_text'][sent]
                if displayed_sents == 3:
                    break
                sent_stats = ', '.join([format_stats(k, sentence_match)
                                        for k in stats])
                render_sentence(sentence_match['text']['content'],
                                sentence_match,
                                state,
                                sentence_match['text']['emb_id'],
                                sent_stats, doc_id,
                                doc_id,  # title_text,
                                section=sent.split(':')[-1].strip("'"))

                displayed_sents += 1

        elif filter['emb_id'] is None or filter['emb_id'] != match_emb_id:
            stats = ', '.join([format_stats(k) for k in stats])
            render_sentence(text, match, state, match_emb_id,
                            doc_title_text, stats, doc_id, html_link)
        # logging.debug(f'display_result({term})',
        #      f'took {time.time()-start} s ')
        return True
    return False


def render_sentence(text, match, state, emb_id, doc_title,
                    stats, doc_id, html_link=None, section=None):
    if not state.result:
        state.result = []
    res = {}
    if section:
        state.outpage.append(f'#### {section}')
        res['section'] = section
    target = text.strip("'")
    state.outpage.append(target.replace('•', '  \n•'))
    res['vänster kontext'], res['träff'],\
        res['höger kontext'] = target.split('**')
    res['stats'] = stats

    if state.debug == 1:
        if len(match['rank']) > 1:
            debug_stats = pd.DataFrame({'rank': match["rank"]
                                        + [sum(match['rank']) /
                                           len(match['rank'])],
                                        'distance': match["distances"]
                                        + [match['distance'] /
                                           len(match['distances'])],
                                        'prompts': match['prompts'] + ['avg']})
        else:
            if state.search_type == 'ämne':
                #print({'rank': match["rank"],
                #                            'distance': match["distance"],
                 #                           'prompt': match['prompts']})
                debug_stats = pd.DataFrame({
                    # 'rank': match["rank"],
                    'distance': match["distances"],
                    'prompt': match['prompts']})
            else:
                debug_stats = pd.DataFrame({
                    # 'rank': match["rank"],
                    'distance': match["distances"]})

        state.outpage.append('  \n'
                             + debug_stats.to_markdown().replace('\n',
                                                                 '  \n'))
        state.outpage.append(f'  \nembedding id: {emb_id},' +
                             f' combined distance: {match["distance"]}')
    res['doc'] = doc_title
    if html_link:
        state.outpage.append(
            f'  \n_Här hittar du dokumentet:_ [{doc_title}]({html_link})')
        res['html'] = html_link
    preset_params = parse.urlencode({'emb_id': emb_id,
                                     'doc_id': doc_id,
                                     'time_from': state.time_from,
                                     'time_to': state.time_to,
                                     'n_results': state.n_results,
                                     'search_type': 'mening',
                                     'scope': state.scope,
                                     'debug': state.debug,
                                     'top_n_ranking': state.top_n_ranking,
                                     'rank_by': state.rank_by,
                                     'selected_model': state.selected_model},
                                    doseq=True)
    state.outpage.append(
        '[visa fler resultat som liknar avsnittet!]' +
        f'(http://localhost:8501/?{preset_params})')

    state.result.append(res)

def order_results_by_sents(distances, neighbours, prompts, text, rank_by):
    start_t = time.time()
    logging.debug('start sorting by sents')
    match_dict = {}
    ranked_dict = OrderedDict()

    def rank_func(x):
        filters = []
        for filter in rank_by:
            # if filter == 'average rank':
            #     filters.append(-sum(match_dict[x]['rank']) /
            #                    len(match_dict[x]['rank']))
            if filter == 'average distance':
                filters.append(1 - match_dict[x]['distance'] /
                               len(match_dict[x]['distances']))
        if not rank_by:
            filters = -1*match_dict[x]['distance']
        return filters

    # compute context size (ignoring document field)
    available_context = len(text[0]) - 1
    end_left = math.ceil(available_context/2)
    for j, n in enumerate(neighbours):
        contents = " ".join([' '.join(text[n][1:end_left]),
                             '**' + text[n][end_left] + '**',
                             ' '.join(text[n][end_left+1:])])
        doc_id, id = text[n][0].split(':', 1)
        embedding_id = n
        key = (contents, doc_id, id, embedding_id)
        if key not in match_dict:
            match_dict[key] = {'rank': [],
                               'original rank': [],
                               'nb_matches': 0,
                               'distance': 0,
                               'distances': [],
                               'prompt_ids': [],
                               'prompts': []}
        for i, prompt in enumerate(prompts):
            match_dict[key]['original rank'].append(j)
            match_dict[key]['distances'].append(distances[i][j])
            match_dict[key]['prompt_ids'].append(i)
            match_dict[key]['prompts'].append(prompt)
        match_dict[key]['distance'] = sum(match_dict[key]['distances'])
    # heuristics to get complete average:
    # we assume each neighbor missing in a prompt's
    # top n ranking is at best at rank n+1 at a similar
    # distance to the nth nearest neighbor

    for key in sorted(match_dict, key=rank_func, reverse=True):
        ranked_dict[key] = match_dict[key]
    logging.debug('stop sorting')
    return ranked_dict, time.time() - start_t


# @st.cache()
# @profile
def fit_nn_model(train, n=40):
    start = time.time()
    nn = NearestNeighbors(n_neighbors=n, metric='cosine', p=1)
    nn.fit(train['embeddings'])
    logging.debug(f'fit_nn_model took {time.time()-start} s')
    return nn


#@st.cache(allow_output_mutation=True)
# @profile
def run_ranking(prompts, train, filter, rank_by, n=30,
                sorting_func=order_results_by_sents, emb_id=None,
                selected_model=0, nn=None):
    logging.info('start run_ranking')
    start = time.time()
    if emb_id is None:
        embeddings = embed_text(prompts, save_out=False, selected_model=selected_model)
    else:
        embeddings = torch.unsqueeze(train['embeddings'][emb_id], dim=0)
    #nn = fit_nn_model(train)
    if len(prompts) > 1:
        i = 5
        prompt = prompts[i]
        reranking_prompts = prompts[:i] + prompts[i+1:]
        prompts = [prompt] + reranking_prompts
        logging.info('call kneighbors')
        top_k_dist, top_k_id = nn.kneighbors(torch.unsqueeze(embeddings[i], axis=0), n_neighbors=n)
        logging.info('finished')
        top_k_id = top_k_id[0]
        top_k_emb = torch.squeeze(torch.stack([train['embeddings'][i] for i in top_k_id]))
        # rerank based on remaining prompts
        reranked_dist = torch.tensor(
            cosine_distances(torch.cat([embeddings[:i],
                                        embeddings[i+1:]]),
                             top_k_emb))
        dist = torch.cat([torch.tensor(top_k_dist), reranked_dist])
    else:
        dist, top_k_id = nn.kneighbors(embeddings, n_neighbors=n)
        dist = torch.tensor(dist)
        top_k_id = top_k_id[0]
    logging.info('end run_ranking')
    logging.debug(f'run_ranking({prompts}) for {n} neighbors ' +
                  f' took {time.time()-start} s ({sorting_func})' +
                  f'{len(top_k_id)} neighbours and {dist.shape}')
    ranking_time = time.time() - start
    return sorting_func(dist, top_k_id, prompts, train['meta'],
                        rank_by), ranking_time

#@st.cache
# @profile
def order_results_by_documents(distances, neighbours, prompts, text, rank_by):
    """
    groups matches by document and orders according to avg document rank and
    similarity (still needs to factor in average match count per document)
    """

    start_t = time.time()
    logging.debug('start sorting')

    def rank_func(x):
        filters = []
        for filter in rank_by:
            if filter == 'count':
                filters.append(match_dict[x]['count'])
            if filter == 'average rank':
                filters.append(-match_dict[x]['rank'])
            if filter == 'average distance':
                filters.append(1-match_dict[x]['distance'])
        if not rank_by:
            filters = -1*match_dict[x]['distance']

        return filters

    match_dict = {}

    # compute context size (ignoring document field)
    available_context = len(text[0]) - 1
    end_left = math.ceil(available_context/2)
    for j, n in enumerate(neighbours):
        match = {}
        contents = " ".join([' '.join(text[n][1:end_left]),
                             '**' + text[n][end_left] + '**',
                             ' '.join(text[n][end_left+1:])])

        match['content'] = contents
        match['doc_id'], id = text[n][0].split(':', 1)
        match['emb_id'] = n
        if match['doc_id'] not in match_dict:
            match_dict[match['doc_id']] = {'rank': 0,
                                           # 'count': 0,
                                           'nb_matches': 0,
                                           'distance': 0,
                                           'distances': [],
                                           'matched_text': {}}
        key = match['doc_id']
        if id not in match_dict[key]['matched_text']:
            match_dict[key]['matched_text'][id] = {
                'rank': [],
                # 'count': 0,
                'distance': 0,
                'distances': [],
                'prompt_ids': [],
                'prompts': [],
                'text': match}
        for i, prompt in enumerate(prompts):
            distance = distances[i][j]
            match_dict[key]['matched_text'][id]['rank'].append(j)
            match_dict[key]['matched_text'][id]['distances'].append(distance)
            # match_dict[key]['matched_text'][id]['count'] += 1
            match_dict[key]['matched_text'][id]['prompt_ids'].append(i)
            match_dict[key]['matched_text'][id]['prompts'].append(prompt)

    for doc_id, neighbor in match_dict.items():
        count = 0
        for text in neighbor['matched_text']:
            neighbor['matched_text'][text]['distance'] = sum(
                neighbor['matched_text'][text]['distances'])
            stats = neighbor['matched_text'][text]
            avg_rank = sum([int(el) + 1 for el in stats['rank']]) / len(
                stats['rank'])
            avg_distance = stats['distance'] / len(stats['distances'])
            # match_dict[doc_id]['count'] += stats['count']
            match_dict[doc_id]['rank'] += avg_rank
            match_dict[doc_id]['distance'] += avg_distance
            match_dict[doc_id]['nb_matches'] += 1
            count += 1
        # match_dict[doc_id]['count'] /= count
        match_dict[doc_id]['rank'] /= count
        match_dict[doc_id]['distance'] /= count
        match_dict[doc_id]['distance'] = match_dict[doc_id]['distance'].item()
    neighbours = OrderedDict()
    for doc in sorted(match_dict,
                      key=rank_func,
                      reverse=True):
        neighbours[doc] = match_dict[doc]
    logging.debug('stop sorting')
    return neighbours, time.time() - start_t


with open('ids2doc.pickle', 'rb') as ifile:
    ids2doc = pickle.load(ifile)


# @profile
def main():
    logging.debug('start main')
    state = _get_state()
    read_query_params(state)
    if not state.train:
        state.train, state.nn = load_documents(
            'matches/match_embeddings.gzip',
            'matches/match_text.csv')

    # Display the search page with the current session state
    page_sent_search(state)

    # Mandatory to avoid rollbacks with widgets,
    # must be called at the end of your app
    # logging.debug('start sync')
    # state.sync()
    # logging.debug('end sync')
    logging.debug('end main')


def automate_tests():
    print('start tests')
    from itertools import product
    logging.root.setLevel(logging.CRITICAL)
    state = _get_state()
    run_stats = {'sorting time':[], 'ranking time': [], 'n_results': [],
                 'term': [], 'n_rank': [], 'rank setting': [], 'grouping': []}
    on_gpu, tokenizer, model = init_ct_model(0)
    # terms = ['klimat', 'hälsa', 'missbruk', 'arbetslöshet', 'tillväxt']
    terms = ['Koldioxidutsläppen leder till klimatförändringar.',
             'Åtskilliga studier visar att pensionering leder till bättre hälsa',
             'All tidigare vård har lett till återfall i missbruk. ”',
             'Det har lett till en överetablering och en permanent arbetslöshet bland konstnärerna.',
             'Framgången i dessa strävanden påverkar både sysselsättning och ekonomisk tillväxt i Sverige.']
    n_ranks = [1000, 10000] #, 100000]
    groupings = [0, 1]
    rank_settings = [[], ['average distance']]
    settings = list(product(terms, n_ranks, groupings, rank_settings))
    with open('settings.csv', 'w') as ofile:
        wr = csv.writer(ofile, delimiter=';')
        for setting in settings:
            wr.writerow(setting)
    if state.seen is None:
        state.seen = []
    print(len(settings))
    n_runs=10
    for i, setting in enumerate(settings):
        print(f'setting {i + 1} of {len(settings)}: {setting}, already seen: {state.seen}')
        if setting in state.seen:
            print('skipping', setting)
            continue
        else:
            state.seen.append(setting)
        query, n_rank, search_grouping, rank_setting = setting
        print()
        print(query, n_rank, search_grouping, rank_setting)
        run_stats['term'].append(query)
        run_stats['n_rank'].append(n_rank)
        run_stats['grouping'].append(search_grouping)
        
        state, new_stats = automate_test_run(state, query, n_rank,
                                             search_grouping, rank_setting,
                                             n_runs=n_runs)
        if rank_setting :
            run_stats['rank setting'].append(rank_setting[0])
        else:
            run_stats['rank setting'].append('default')
        run_stats['sorting time'].append(new_stats['sorting time'])
        run_stats['ranking time'].append(new_stats['ranking time'])
        run_stats['n_results'].append(new_stats['n_results'])
        print(run_stats)

    print('done?')
    run_stats = pd.DataFrame(run_stats)
    if state.run_stats is None:
        state.run_stats = run_stats
    else:
        state.run_stats = pd.concat([pd.DataFrame(state.run_stats), run_stats])
    state.run_stats.to_excel(f'sents_test_{n_runs}_runs_statistics.xlsx', engine='xlsxwriter')
    print(state.run_stats)
    print('end tests')
    print('start sync')
    state.sync()
    print('end sync')

    sys.exit()


# @profile
def automate_test_run(state, query, n_rank, search_grouping, rank_setting, n_runs=2):
    logging.debug('start test run')
    read_query_params(state)
    emb_id = None
    state.time_from = 1994
    state.time_to = 2020
    if ' ' in query:
        state.query = query
        state.search_type = 'mening'
    else:
        state.query_effect = query
        state.search_type = 'ämne'
    state.top_n_ranking = n_rank
    state.scope = search_grouping
    state.rank_by = rank_setting
    state.run_stats = {'sorting time':[], 'ranking time': [], 'n_results': []}
    if not state.train:
        state.train = load_documents(
            'matches/match_embeddings.gzip',
            'matches/match_text.csv')
    for _ in tqdm(range(n_runs)):
        state.start_search = True
        #page_sent_search(state)
        state.outpage = []
        state.outpage.append(
            f'# Resultat för {state.search_type}sbaserat sökning')
        if (state.query_cause or state.query_effect)\
           and state.search_type == 'ämne':
            state.query = None
            prompts = generate_prompts(cause=state.query_cause,
                                       effect=state.query_effect)
            query = ''
            if state.query_cause:
                query += f'orsak:  “{state.query_cause}”'
            if state.query_effect:
                query += f', verkan:  “{state.query_effect}”'
            query = query.lstrip(', ')
            state.outpage.append(f'## _{query}_')
            rank(state, prompts, emb_id=emb_id)
        elif state.query:
            state.query_cause = state.query_effect = None
            query = state.query
            if not emb_id:
                state.outpage.append(f'##  “_{query}_”')
            rank(state, [state.query], emb_id=emb_id)
        #st.write(state.result)
        if state.result:
            table = pd.DataFrame(state.result)
            if state.run_stats is not None:
                state.run_stats['n_results'].append(len(table))
            state.outpage = state.outpage[:state.insert_link]\
                + [get_table_download_link(table, query)]\
                + state.outpage[state.insert_link:]
        #st.markdown('  \n'.join(state.outpage[:-1]), unsafe_allow_html=True)

    # Mandatory to avoid rollbacks with widgets,
    # must be called at the end of your app
    run_stats = {k:sum(v)/len(v) for k,v in state.run_stats.items()}
    print(run_stats)
    logging.debug('end test run')
    return state, run_stats


def setup_settings_bar(state):
    st.sidebar.title(":wrench: Inställningar")
    # this is buggy use button instead?
    # if st.sidebar.checkbox('begränsa träffmängden'):
    #     if not state.n_results:
    #         state.n_results = 10
    #         state.n_results = st.sidebar.slider('max antal matchningar',
    #                                             min_value=1, max_value=30,
    #                                             value=state.n_results)
    # else:
    state.n_results = 0

    # st.sidebar.markdown('---')
    # st.sidebar.markdown('## Model')
    state.model_options = {'simple ct': "Contrastive-Tension/BERT-Base-Swe-CT-STSb",
                           'custom ct': "Uppsala-Relations-Model-HumanStop.weights"}
    select_options = sorted(state.model_options, reverse=True)
    if state.selected_model is None:
        state.selected_model = 0
    # state.selected_model = st.sidebar.radio('specify model for nn ranking',
    #                                         select_options,
    #                                         state.selected_model)
    # state.selected_model = select_options.index(state.selected_model)
    state.rerank_by = None  # add new model as filter
    # st.sidebar.markdown('---')
    if not state.top_n_ranking:
        state.top_n_ranking = 10
    state.top_n_ranking = st.sidebar.number_input('Top n ranking:',
                                                  min_value=5,
                                                  value=state.top_n_ranking)
    # if st.sidebar.checkbox('unlimited'):
    #    state.top_n_ranking = len(state.train['embeddings'])

    st.sidebar.markdown('---')
    st.sidebar.markdown('## Sökfråga')
    select_options = ['ämne', 'mening']
    index = select_options.index(state.search_type) \
        if state.search_type else 0
    state.search_type = st.sidebar.radio('söka efter', select_options, index)

    st.sidebar.markdown('---')
    st.sidebar.markdown('## Resultat')
    select_options = ['average distance', 'summed distance']  # ['average rank', 'average distance']
    state.rank_by = st.sidebar.multiselect('rangordna efter', select_options,
                                           state.rank_by)
    select_options = ['enskilda meningar', 'dokument']
    index = state.scope if state.scope else 0
    state.scope = select_options.index(st.sidebar.radio('gruppering',
                                                        select_options, index))

    st.sidebar.markdown('---')
    st.sidebar.markdown('## Filter')
    from_options = [i for i in range(1994, 2020, 1)]
    index = 0
    if state.time_from:
        index = from_options.index(state.time_from)
    state.time_from = st.sidebar.selectbox('fr.o.m', from_options, index=index)
    to_options = sorted([el for el in from_options if el > state.time_from],
                        reverse=True)
    if state.time_to:
        index = to_options.index(state.time_to)
    state.time_to = st.sidebar.selectbox('t.o.m', to_options, index=index)

    st.sidebar.markdown('---')
    select_options = ['off', 'on']
    index = state.debug if state.debug else 0
    state.debug = select_options.index(st.sidebar.radio('Debug',
                                                        select_options, index))
    # update_query_params(state)


def read_default_params(state):
    """
    read previous parameters from state
    or return default
    """
    logging.debug('reading parameters')
    default = ''
    cause_default = ''
    effect_default = ''
    emb_id = None
    query_params = get_query_params(state)
    if state.debug:
        st.write(f'QUERY params: {query_params}')
        st.write(f'using model: {state.selected_model}')
    if 'emb_id' in query_params:
        emb_id = query_params['emb_id']
        if isinstance(emb_id, list):
            emb_id = emb_id[0]
        if isinstance(emb_id, str):
            emb_id = int(emb_id)
        default = state.train['meta'][emb_id][3]
        state.search_type = 'mening'
    else:
        if state.query is not None:
            default = state.query
        if state.query_cause is not None:
            cause_default = state.query_cause
        if state.query_effect is not None:
            effect_default = state.query_effect
    return default, cause_default, effect_default, emb_id
    # return '', '', '', emb_id


def page_sent_search(state):
    print('in page sent search')
    setup_settings_bar(state)
    default, cause_default, effect_default, emb_id = read_default_params(state)
    if emb_id is not None:
        # there is a bug/undesired refreshing of the page that interferes here!
        # sometimes?
        st.title(":mag: Fler resultat som ...")
        st.markdown(f'##  “_{default}_”')
        start_search = True
    else:
        st.title(":mag: Sökning")
        if state.search_type == 'mening':
            state.query = st.text_input('Ange en mening',
                                        default)
        else:
            state.query_cause = st.text_input('Ange en orsak', cause_default)
            state.query_effect = st.text_input('Ange en effekt',
                                               effect_default)

        update_query_params(state)
        start_search = st.button('skapa ny sökfråga')
    if start_search and\
       (((state.query not in [None, ''] or emb_id)
         and state.search_type == 'mening')
        or ((state.query_cause not in [None, '']
             or state.query_effect not in [None, ''])
            and state.search_type == 'ämne')):
        state.outpage = []
        query = ''
        state.outpage.append(
            f'# Resultat för {state.search_type}sbaserat sökning')
        if (state.query_cause or state.query_effect)\
           and state.search_type == 'ämne':
            state.query = None
            prompts = generate_prompts(cause=state.query_cause,
                                       effect=state.query_effect)
            query = ''
            if state.query_cause:
                query += f'orsak:  “{state.query_cause}”'
            if state.query_effect:
                query += f', verkan:  “{state.query_effect}”'
            query = query.lstrip(', ')
            state.outpage.append(f'## _{query}_')
            rank(state, prompts, emb_id=emb_id)
        elif default:
            state.query_cause = state.query_effect = None
            query = default
            if not emb_id:
                state.outpage.append(f'##  “_{query}_”')
            rank(state, [default], emb_id=emb_id)
        if state.result:
            table = pd.DataFrame(state.result)
            if state.run_stats is not None:
                state.run_stats['n_results'].append(len(table))
            state.outpage = state.outpage[:state.insert_link]\
                + [get_table_download_link(table, query)]\
                + state.outpage[state.insert_link:]
        st.markdown('  \n'.join(state.outpage[:-1]), unsafe_allow_html=True)
    elif state.outpage:
        st.markdown('  \n'.join(state.outpage[:-1]), unsafe_allow_html=True)

# @profile
def rank(state, prompts, emb_id=None):
    state.outpage.append('---')
    state.insert_link = len(state.outpage) - 1
    # reset results
    state.result = []
    logging.debug('start ranking')
    start = time.time()
    if not hasattr(state, 'train') or not state.train:
        state.train = load_documents(
            'matches/match_embeddings.gzip',
            'matches/match_text.csv')
    if isinstance(prompts, str):
        term = prompts
        prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) == 1:
        term = prompts[0]
    else:
        term = ''
        if state.query_cause:
            term += f'Orsak: {state.query_cause}'
        if state.query_effect:
            term += f'; Verkan: {state.query_effect}'
        term = term.strip('; ')
        state.term = term
    ranking_key = (term, state.scope, state.top_n_ranking,
                   ' '.join(state.rank_by))

    # if not state.ranking:
    #     state.ranking = {}
    # ranking_key = (term, state.scope, state.top_n_ranking,
    #                ' '.join(state.rank_by))
    # if ranking_key not in state.ranking:
    if True:
        logging.debug(f'ranking {ranking_key}')
        sorting_func = order_results_by_documents if state.scope == 1\
            else order_results_by_sents
        (ranking, sorting_time), ranking_time = run_ranking(
            prompts, state.train,
            filter={'time_from': state.time_from,
                    'time_to': state.time_to,
                    'emb_id': emb_id},
            rank_by=state.rank_by,
            n=state.top_n_ranking, emb_id=emb_id,
            sorting_func=sorting_func,
            selected_model=state.selected_model,
            nn=state.nn)
        # keep track of last ranking
        state.ranking_key = ranking_key
        if state.run_stats is not None:
            state.run_stats['sorting time'].append(sorting_time)
            state.run_stats['ranking time'].append(ranking_time)
    n_matches = 0
    ranking_unit = 'document' if state.scope == 1 else 'sentence'
    logging.info(
        f'ranking {len(ranking)}' +
        f' {ranking_unit}s for "{ranking_key}"')
    seen_documents = []
    for el in ranking:
        hit = display_result(state, term, el, {'time_from': state.time_from,
                                               'time_to': state.time_to,
                                               'emb_id': emb_id},
                             seen_documents,
                             match=ranking[el]
)
        if hit:
            if isinstance(el, tuple):
                text, doc_id, sent_nb, match_emb_id = el
            else:
                doc_id = el
            doc_id = doc_id.split('_')[-1].split('.')[0]
            seen_documents.append(doc_id)
            n_matches += 1
            if state.n_results and n_matches >= state.n_results:
                break
            state.outpage.append('---')
    logging.debug(f'ranking({prompts}) took {time.time()-start} s ')
    logging.debug('all matches displayed')


def read_query_params(state=None):
    """
    retrieve url query parameters and update the state
    or return them.
    """
    if not state:
        state = {}
    for k, v in st.experimental_get_query_params().items():
        if v[0].isnumeric():
            state[k] = int(v[0])
        else:
            if len(v) > 1:
                state[k] = v
            else:
                state[k] = v[0] if not v[0] == 'None' else None
    if isinstance(state, dict):
        return state


def update_query_params(state):
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect', 'top_n_ranking',
              'rank_by', 'selected_model']
    updated_states = {}

    for param in params:
        if state[param]:
            updated_states[param] = state[param]

    st.experimental_set_query_params(**updated_states)


def get_query_params(state):
    query_params = st.experimental_get_query_params()
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect', 'top_n_ranking',
              'rank_by']
    for p in params:
        if p not in query_params and state[p] is not None:
            query_params[p] = state[p]
    return query_params


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """
        Rerun the app with all state values up to date
        from the beginning to fix rollbacks.
        """

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                    self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()
    #st.stop()
    # automate_tests()
