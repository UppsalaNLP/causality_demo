import csv
import base64
import time
# from tqdm import tqdm
import sys
from io import BytesIO
import pickle
from collections import OrderedDict
import gzip
import logging
import math
import os
import re
from typing import List, Dict, Tuple
from urllib import parse

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.server.server import Server
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer,\
    BertModel
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
# from memory_profiler import profile

from multipage_session_state import _SessionState as SessionState
from multipage_session_state import _get_state

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO)
logging.root.setLevel(logging.INFO)
# logging.root.setLevel(logging.NOTSET)

st.set_page_config(page_title='demo app',
                   page_icon=':mag:',
                   layout='centered',
                   initial_sidebar_state='expanded')


def get_table_download_link(table: pd.DataFrame, query: str) -> str:
    """
    save matches and meta data stored in table as excel file
    :param table: a data frame of matches and corresponding meta data
    :param query: a string representation of the query terms and
                  respective causal roles
    :return: a string containing mark up of the download url
    """

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


@st.cache(show_spinner=False)
def generate_prompts(cause: str = None, effect: str = None) -> List[str]:
    """
    insert cause and/or effect terms into prompt templates.
    :return: a list of completely or partially filled prompts
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


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_binary(emb_file: str) -> torch.Tensor:
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


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_documents(input_emb: str,
                   input_meta: str) -> Dict[str, torch.Tensor]:
    """
    load prefiltered text and embeddings.
    :return: a dictionary of text and embeddings
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
    return docs


@st.cache(allow_output_mutation=True, show_spinner=False)
def init_ct_model() -> Tuple[bool, PreTrainedTokenizer,
                             BertModel]:

    logging.debug('loading model')
    on_gpu = torch.cuda.is_available()
    logging.debug(f'GPU available: {on_gpu}')
    model_name = tok_name = "Contrastive-Tension/BERT-Base-Swe-CT-STSb"
    logging.debug(f'loading tokeniser: {tok_name}')
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    logging.debug(f'loading BERT model: {model_name}')
    model = AutoModel.from_pretrained(model_name, from_tf=True)
    return on_gpu, tokenizer, model


@st.cache(show_spinner=False)
def embed_text(samples: List[str]) -> torch.Tensor:
    """
    embed samples using the Swedish STS model
    """
    from torch.utils.data import DataLoader, SequentialSampler
    on_gpu, tokenizer, model = init_ct_model()
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
    logging.debug('done')
    return embeddings


def mean_pool(model_out: torch.Tensor,
              input_mask: torch.Tensor) -> torch.Tensor:
    """
    apply mean pooling to the word embeddings of the output layer
    following sentence_transformers
    """

    embeddings = model_out[0]
    attention_mask = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * attention_mask, 1)
    n = torch.clamp(attention_mask.sum(1), min=1e-9)
    return sum_embeddings / n


def display_result(state: SessionState, term: str, doc_id: str,
                   filter: Dict[str, int], seen_documents: List[str],
                   match: Dict) -> bool:
    """
    render a single match if it matches the filter
    the match is appended to state.outpage
    """
    if isinstance(doc_id, tuple):
        text, doc_id, sent_nb, match_emb_id = doc_id
    doc_id = doc_id.split('_')[-1].split('.')[0]

    if doc_id in seen_documents:
        return False
    doc_title, date = ids2doc[doc_id]

    year = doc_title.split()[-1].split(':')[0]
    assert year.isnumeric(), f'malformed document title {doc_title} ({year})'
    year = int(year)

    def rank_func(x):
        return -1*match['matched_text'][x]['distance']

    def format_stats(k, match=match):
        return (": ".join(
            [k, f"{match[k]:>1.3f}"])
                if isinstance(match[k], float)
                else ": ".join([k,
                                f"{match[k]}"
                                if isinstance(match[k], list)
                                else str(match[k])]))\
                                     if k in match else ''

    if year in range(filter['time_from'], filter['time_to'] + 1)\
       and (state.doc_id is None or state.doc_id != doc_id):

        displayed_sents = 0
        doc_title_text = doc_title
        continuation = re.findall(r'd(\d+)$', doc_id)
        html_link = f'https://data.riksdagen.se/dokument/{doc_id}'
        if continuation:
            doc_title_text = f'{doc_title}, del {continuation[0]}'

        if 'matched_text' in match.keys():
            state.outpage.append(f'## [{doc_title_text}]({html_link})')
            stats_header = f'_avstånd: {match["distance"]:>1.3f}_'
            state.outpage.append(stats_header)
            nb_matches = len(match['matched_text'])
            to_display = nb_matches if state.doc_results else 3
            if nb_matches > 1:
                state.outpage.extend(
                    ['### __Bästa resultat__',
                     f'''(_visar {min(to_display, nb_matches)} av
                     {nb_matches} träff_)'''])
            for sent in sorted(match['matched_text'],
                               key=rank_func, reverse=True):
                sentence_match = match['matched_text'][sent]
                if displayed_sents == to_display:
                    break
                sent_stats = f'{sentence_match["distance"]:>1.3f}'
                render_sentence(sentence_match['text']['content'],
                                sentence_match,
                                state,
                                emb_id=sentence_match['text']['emb_id'],
                                doc_title=doc_title_text,
                                stats=sent_stats,
                                doc_id=doc_id,
                                section=sent.split(':')[-1].strip("'"))

                displayed_sents += 1

        elif filter['emb_id'] is None or filter['emb_id'] != match_emb_id:
            stats = f'{match["distance"]:>1.3f}'
            render_sentence(text, match, state, match_emb_id,
                            doc_title_text, stats, doc_id, html_link)
        return True
    return False


def render_sentence(text: str, match, state: SessionState, emb_id: int,
                    doc_title: str, stats: str, doc_id: str,
                    html_link: str = None, section: str = None):
    if not state.result:
        state.result = []
    res = {}
    if section:
        state.outpage.append(f'### {section}')
        res['section'] = section
    target = text.strip("'")
    state.outpage.append(target.replace('•', '  \n•'))
    res['vänster kontext'], res['träff'],\
        res['höger kontext'] = target.split('**')
    res['combined distance'] = stats

    # if state.debug == 1:
    #     if len(match['distances']) > 1:
    #         debug_stats = pd.DataFrame({'distance': match["distances"]
    #                                     + [match['distance'] /
    #                                        len(match['distances'])],
    #                                     'prompts': match['prompts'] + ['avg']
    # })
    #     else:
    #         if state.search_type == 'ämne':
    #             debug_stats = pd.DataFrame({
    #                 'distance': match["distances"],
    #                 'prompt': match['prompts']})
    #         else:
    #             debug_stats = pd.DataFrame({
    #                 'distance': match["distances"]})

    #     state.outpage.append('  \n'
    #                          + debug_stats.to_markdown().replace('\n',
    #                                                              '  \n'))
    #     state.outpage.append(f'  \nembedding id: {emb_id},' +
    #                          f' combined distance: {match["distance"]}')
    res['doc'] = doc_title
    if html_link:
        state.outpage.append(f'  \n_avstånd: {match["distance"]:>1.3f}_')
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
                                     'doc_results': state.doc_results,
                                     'unique_docs': state.unique_docs,
                                     'top_n_ranking': state.top_n_ranking},
                                    doseq=True)
    state.outpage.append(
        '  \n[visa fler resultat som liknar avsnittet!]' +
        f'(http://{get_host()}/?{preset_params})')

    state.result.append(res)


def order_results_by_sents(distances: torch.Tensor,
                           neighbours: np.ndarray,
                           prompts: List[str],
                           text: List[str]) -> OrderedDict:
    logging.debug('start sorting by sents')
    match_dict = {}
    ranked_dict = OrderedDict()

    def rank_func(x):
        return -1*match_dict[x]['distance']

    for j, n in enumerate(neighbours):
        contents = " ".join([' '.join(text[n][1:3]),
                             '**' + text[n][3] + '**',
                             ' '.join(text[n][4:])])
        doc_id, id = text[n][0].split(':', 1)
        embedding_id = n
        key = (contents, doc_id, id, embedding_id)
        if key not in match_dict:
            match_dict[key] = {'original rank': [],
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

    for key in sorted(match_dict, key=rank_func, reverse=True):
        ranked_dict[key] = match_dict[key]
    logging.debug('stop sorting')
    return ranked_dict


def fit_nn_model(embeddings: torch.Tensor,
                 n: int = 40) -> NearestNeighbors:
    start = time.time()
    nn = NearestNeighbors(n_neighbors=n, metric='cosine', p=1)
    nn.fit(embeddings)
    logging.debug(f'fit_nn_model took {time.time()-start} s')
    return nn


def run_ranking(prompts: List[str], train: Dict[str, torch.Tensor],
                filter: Dict[str, int], n: int = 30,
                sorting_func=order_results_by_sents, emb_id: int = None,
                nn: NearestNeighbors = None):
    logging.debug('start run_ranking')
    start = time.time()
    if emb_id is None:
        prompt_embeddings = embed_text(prompts)
    else:
        prompt_embeddings = torch.unsqueeze(train['embeddings'][emb_id], dim=0)
    if len(prompts) > 1:
        i = 5
        prompt = prompts[i]
        reranking_prompts = prompts[:i] + prompts[i+1:]
        prompts = [prompt] + reranking_prompts
        logging.debug('call kneighbors')
        top_k_dist, top_k_id = nn.kneighbors(
            torch.unsqueeze(prompt_embeddings[i], axis=0),
            n_neighbors=n)
        logging.debug('finished')
        top_k_id = top_k_id[0]
        top_k_emb = torch.squeeze(torch.stack([train['embeddings'][i]
                                               for i in top_k_id]))
        # rerank based on remaining prompts
        reranked_dist = torch.tensor(
            cosine_distances(torch.cat([prompt_embeddings[:i], prompt_embeddings[i+1:]]),
                             top_k_emb))
        dist = torch.cat([torch.tensor(top_k_dist), reranked_dist])
    else:
        dist, top_k_id = nn.kneighbors(prompt_embeddings, n_neighbors=n)
        dist = torch.tensor(dist)
        top_k_id = top_k_id[0]
    logging.debug('end run_ranking')
    logging.debug(f'run_ranking({prompts}) for {n} neighbors ' +
                  f' took {time.time()-start} s ({sorting_func})' +
                  f'{len(top_k_id)} neighbours and {dist.shape}')
    return sorting_func(dist, top_k_id, prompts, train['meta'])


def order_results_by_documents(distances: torch.Tensor,
                               neighbours: np.ndarray,
                               prompts: List[str],
                               text: List[str]) -> OrderedDict:
    """
    groups matches by document and orders according to avg document rank and
    similarity (still needs to factor in average match count per document)
    """

    logging.debug('start sorting')

    def rank_func(x):
        return -1*match_dict[x]['distance']

    match_dict = {}

    for j, n in enumerate(neighbours):
        match = {}
        contents = " ".join([' '.join(text[n][1:3]),
                             '**' + text[n][3] + '**',
                             ' '.join(text[n][4:])])

        match['content'] = contents
        match['doc_id'], id = text[n][0].split(':', 1)
        match['emb_id'] = n
        if match['doc_id'] not in match_dict:
            match_dict[match['doc_id']] = {'nb_matches': 0,
                                           'distance': 0,
                                           'distances': [],
                                           'matched_text': {}}
        key = match['doc_id']
        if id not in match_dict[key]['matched_text']:
            match_dict[key]['matched_text'][id] = {
                'distance': 0,
                'distances': [],
                'prompt_ids': [],
                'prompts': [],
                'text': match}
        for i, prompt in enumerate(prompts):
            distance = distances[i][j]
            match_dict[key]['matched_text'][id]['distances'].append(distance)
            match_dict[key]['matched_text'][id]['prompt_ids'].append(i)
            match_dict[key]['matched_text'][id]['prompts'].append(prompt)

    for doc_id, neighbor in match_dict.items():
        count = 0
        for text in neighbor['matched_text']:
            neighbor['matched_text'][text]['distance'] = sum(
                neighbor['matched_text'][text]['distances'])
            stats = neighbor['matched_text'][text]
            # avg_distance = stats['distance'] / len(stats['distances'])
            summed_distance = stats['distance']
            match_dict[doc_id]['distance'] += summed_distance
            match_dict[doc_id]['nb_matches'] += 1
            count += 1
        match_dict[doc_id]['distance'] /= count
        match_dict[doc_id]['distance'] = match_dict[doc_id]['distance'].item()
    neighbours = OrderedDict()
    for doc in sorted(match_dict,
                      key=rank_func,
                      reverse=True):
        neighbours[doc] = match_dict[doc]
    logging.debug('stop sorting')
    return neighbours


with open('ids2doc.pickle', 'rb') as ifile:
    ids2doc = pickle.load(ifile)


def main(data='full_sou'):
    logging.debug('start main')
    state = _get_state()
    state.data = data
    read_query_params(state)

    # display the search page with the current session state
    page_sent_search(state)

    # mandatory to avoid rollbacks with widgets
    logging.debug('start sync')
    state.sync()
    logging.debug('end sync')
    logging.debug('end main')


def setup_settings_bar(state: SessionState):
    st.sidebar.title(":wrench: Inställningar")

    state.n_results = 0

    st.sidebar.markdown(
        '''
        ## Sökfråga

        Systemet tar olika typer av input.

        Man kan antingen ge en full __mening__ eller fråga,
        t.ex.

        * _"Stress leder till sjukskrivningar."_
        * _"Vad orsakar arbetslöshet?"_

        eller nyckelord/__ämnen__ för en orsak och/eller verkan, t.ex.

        * effekt: _"stigande räntor"_
        * orsak: _"skattesänkningar"_
        * orsak: _"jordbruk"_, effekt: _"övergödning"_.
        ''')
    select_options = ['mening', 'ämne']
    index = select_options.index(state.search_type) \
        if state.search_type else 0
    state.search_type = st.sidebar.radio('söka efter', select_options, index)

    st.sidebar.markdown(
        """
        ## Rankning

        Bestämmer hur många träff modellen tar hänsyn till -
        t.ex. bara de 50 eller 100 mest linknande meningar.

        Ju fler träffar desto längre tid tar rankningen.
        """)
    if not state.top_n_ranking:
        state.top_n_ranking = 100
    state.top_n_ranking = st.sidebar.number_input('Top n rankning',
                                                  min_value=50,
                                                  max_value=300,
                                                  value=state.top_n_ranking)
    # st.sidebar.markdown('---')

    # st.sidebar.markdown('---')
    st.sidebar.markdown(
        '''
        ## Resultat

        Fastställer hur resultatet ska presenteras.

        Antingen sorteras meningarna efter __dokument__ där dokumentet med
        de i genomsnitt närmaste träffarna är högst eller efter __enskilda
        meningar__, dvs. den närmaste meningen oavsett dokumentet är högst.
        ''')
    select_options = ['dokument', 'enskilda meningar']
    index = state.scope if state.scope else 0
    state.scope = select_options.index(st.sidebar.radio('gruppering',
                                                        select_options, index))
    st.sidebar.markdown('### ytterligare inställningar')
    doc_options = st.sidebar.empty()
    if state.scope:
        state.unique_docs = doc_options.checkbox(
            'visa bara bästa träffen per dokument',
            value=False)
        st.sidebar.markdown(''':arrow_right: om omarkerad visar systemet alla n meningar
        som är närmast sökfrågan''')
    else:
        select_options = ['upp till 3', 'alla']
        index = state.doc_results if state.doc_results else 0
        state.doc_results = select_options.index(
            doc_options.radio('Hur många träff per dokument ska visas?',
                              select_options,
                              index))
    # st.sidebar.markdown('---')
    st.sidebar.markdown(
        '''
        ## Filter

        Begränsar resultatmängden efter tid.
        ''')
    from_options = [i for i in range(1994, 2021, 1)]
    index = 0
    if state.time_from:
        index = from_options.index(state.time_from)
    state.time_from = st.sidebar.selectbox('fr.o.m', from_options, index=index)
    to_options = sorted([el for el in from_options if el > state.time_from],
                        reverse=True)
    if state.time_to:
        index = to_options.index(state.time_to)
    state.time_to = st.sidebar.selectbox('t.o.m', to_options, index=index)

    # st.sidebar.markdown('---')
    # select_options = ['off', 'on']
    # index = state.debug if state.debug else 0
    # state.debug = select_options.index(st.sidebar.radio('Debug',
    #                                                     select_options,
    # index))


def read_default_params(state: SessionState) -> Tuple[str, int]:
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
        if state.data == 'full_sou':
            train = load_documents(
                'matches/match_embeddings.gzip',
                'matches/match_text.csv')
        else:
            # load summary text
            train = load_documents(
                'matches/summary_47477_embeddings.gzip',
                'matches/summary_text.csv')
        default = train['meta'][emb_id][3]
        state.search_type = 'mening'
    else:
        if state.query is not None:
            default = state.query
        if state.query_cause is not None:
            cause_default = state.query_cause
        if state.query_effect is not None:
            effect_default = state.query_effect
    return default, cause_default, effect_default, emb_id


def page_sent_search(state: SessionState):
    setup_settings_bar(state)
    default, cause_default, effect_default, emb_id = read_default_params(state)
    if emb_id is not None:
        st.title(":mag: Fler resultat som ...")
        st.markdown(f'##  “_{default}_”')
        start_search = True
    else:
        st.title(":mag: Sökning")
        st.markdown(
            """
            ## Så här gör du
            Anpassa inställningar i sidofält-menyn, skriv in din sökfråga och
            bekräfta genom att trycka på
            __skapa ny sökfråga__.
            """)
        with st.beta_expander('Hur funkar systemet?'):
            st.markdown(
                '''
Systemet beräknar likheten mellan en sökfråga och en mängd kausala meningar
i Statens offentliga utredningar (SOU) med hjälp av en språkmodell för
 semantisk likhet. Beroende på typen av sökfråga och resultatsgruppering krävs
 det ytterligare rankningar:

* om sökfrågan är en specifik mening jämföras bara likheten mellan meningen och
de olika meningar i SOU:er
* är sökfrågan ett eller fler nyckelord, genereras olika meningar eller fraser
 med ett flertal kausala uttryck (t.ex. _"rökning **leder till** cancer"_ eller
_"cancer **på grund av** rökning"_) extraheras de _n_ bästa träff för
 en av dessa meningar, den genomsnittliga likheten av träffarna till alla
 sökfraser beräknas och träffarna rankas därefter.
* om resultatet grupperas i **dokument** rankas dokumenten efter den
 genomsnittliga avstånd av alla träffar i samma dokument till sökfrågan.
                ''')
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
            f'## __Resultat för {state.search_type}sbaserat sökning__')
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


def rank(state: SessionState, prompts: List[str],
         emb_id: int = None):
    state.outpage.append('---')
    state.insert_link = len(state.outpage) - 1

    # reset results
    state.result = []
    logging.debug('start ranking')
    start = time.time()
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
    ranking_key = (term, state.scope, state.top_n_ranking)
    if state.data == 'full_sou':
        train = load_documents(
            'matches/match_embeddings.gzip',
            'matches/match_text.csv')
    else:
        # load summary text
        train = load_documents(
            'matches/summary_47477_embeddings.gzip',
            'matches/summary_text.csv')
    nn = fit_nn_model(train['embeddings'])
    logging.debug(f'ranking {ranking_key}')
    sorting_func = order_results_by_documents if state.scope == 0\
        else order_results_by_sents
    ranking = run_ranking(
        prompts, train,
        filter={'time_from': state.time_from,
                'time_to': state.time_to,
                'emb_id': emb_id},
        n=state.top_n_ranking, emb_id=emb_id,
        sorting_func=sorting_func,
        nn=nn)

    # keep track of last ranking
    state.ranking_key = ranking_key
    n_matches = 0
    ranking_unit = 'document' if state.scope == 0 else 'sentence'
    logging.debug(
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
            if state.unique_docs:
                seen_documents.append(doc_id)
            n_matches += 1
            if state.n_results and n_matches >= state.n_results:
                break
            state.outpage.append('---')
    logging.debug(f'ranking({prompts}) took {time.time()-start} s ')
    logging.debug('all matches displayed')


def read_query_params(state: SessionState):
    """
    retrieve url query parameters and update the state
    """
    for k, v in st.experimental_get_query_params().items():
        if v[0].isnumeric():
            state[k] = int(v[0])
        else:
            if len(v) > 1:
                state[k] = v
            else:
                state[k] = v[0] if not v[0] == 'None' else None


def update_query_params(state: SessionState):
    """
    update url query parameters based on current session state
    """
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect', 'doc_results',
              'umique_docs', 'top_n_ranking']
    updated_states = {}

    for param in params:
        if state[param]:
            updated_states[param] = state[param]

    st.experimental_set_query_params(**updated_states)


def get_query_params(state: SessionState) -> Dict:
    query_params = st.experimental_get_query_params()
    params = ['emb_id', 'time_from', 'time_to', 'debug',
              'n_results', 'search_type', 'scope', 'query',
              'query_cause', 'query_effect',  'doc_results',
              'umique_docs', 'top_n_ranking']
    for p in params:
        if p not in query_params and state[p] is not None:
            query_params[p] = state[p]
    return query_params


def get_host():
    # Hack to get the session object from Streamlit.

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    # Multiple Session Objects?
    for session_info in session_infos:
        headers = session_info.ws.request.headers
        return headers['Host']


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[-1] == 'full-text':
            main('full_sou')
        elif sys.argv[-1] == 'summary':
            main('summary')
        else:
            print(sys.argv)
            print("""
Unkown input data option!
select 'full-text' to search on SOU fulltext
or 'summary' to search on summaries only
            """)
            sys.exit(1)
    else:
        main()
