from collections import OrderedDict
import sys
import gzip
import csv
import json
import logging
import math
import pickle
import random
import time

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoModel, AutoTokenizer, BertModel,\
    PreTrainedTokenizerFast

sys.path.append('/Users/luisedu/Documents/Projekt/Causality/causality_code/evaluation')
from binary_evaluation import index_sentences, read_html
#sys.path.append('/Users/luidu652/Documents/causality_extraction')
#from generate_eval_data import init_ct_model
# path = '/Users/luidu652/Documents/causality_extraction/'
# toy data


def load_binary(emb_file: str) -> torch.Tensor:
    logging.debug(f'loading {emb_file}')
    start = time.time()
    if emb_file.endswith('.pickle') or emb_file.endswith('.gzip'):
        with open(emb_file, 'rb') as ifile:
            embeddings = pickle.load(ifile)
    else:
        raise RuntimeError(f'unknown file type {emb_file}')
    logging.debug(f'load_binary() took {time.time()-start} s ')
    return embeddings


def load_documents(input_emb: str,
                   input_meta: str) -> Tuple[str, torch.Tensor]:
    """
    load prefiltered text and embeddings.
    :return: a dictionary of text and embeddings
    """
    start = time.time()
    logging.debug('loading embeddings')
    embeddings = load_binary(input_emb)
    if input_meta.endswith('.gz'):
        meta = load_binary(input_meta)
    else:
        with open(input_meta) as ifile:
            reader = csv.reader(ifile, delimiter=';')
            meta = [line for line in reader]
    logging.debug(f'load_documents() took {time.time()-start} s ')
    return embeddings, meta


def init_ct_model(selected_model=0):

    logging.debug('loading model')
    on_gpu = torch.cuda.is_available()
    logging.debug(f'GPU available: {on_gpu}')
    model_name = tok_name = "Contrastive-Tension/BERT-Base-Swe-CT-STSb"
    if selected_model != 0:
        # use new model
        model_name = selected_model
    logging.debug(f'loading tokeniser: {tok_name}')
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    logging.debug(f'loading BERT model: {model_name}')
    model = AutoModel.from_pretrained(model_name, from_tf=True)
    return {'on_gpu': on_gpu, 'tokenizer': tokenizer, 'model':  model}


def generate_prompts(cause: str = None, effect: str = None,
                     restrict_to_prompt: List[str] = None) -> List[str]:
    """
    insert cause and/or effect terms into prompt templates.
    :param cause: a string specifying a cause
    :param effect: a string specifying an effect
    :param restrict_to_prompt: one ore more of the keywords in prompt_dict
    to use for generation if we do not use the full dictionary
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
        'vålla': ['[MASK] vållar X']
    }
    if restrict_to_prompt is not None:
        prompt_dict = {k: prompt_dict[k] for k in restrict_to_prompt}
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


def embed_text(model: BertModel, tokenizer: PreTrainedTokenizerFast,
               on_gpu: bool, samples: List[str]) -> torch.Tensor:
    """
    embed samples using the Swedish STS model
    """
    from torch.utils.data import DataLoader, SequentialSampler
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


def select_target(sentences):
    return [sentence[1] for sentence in sentences]


def rank_selected_sentences(ct_model, sentences, query, prompts_to_keep=None):
    targets = select_target(sentences)
    sent_embeddings = embed_text(**ct_model, samples=targets)
    prompts = generate_prompts(**query, restrict_to_prompt=prompts_to_keep)
    embedded_prompts = embed_text(**ct_model, samples=prompts)
    cosine_dists = cosine_distances(embedded_prompts, sent_embeddings)
    averaged_cosine = [float(el) for el in cosine_dists.sum(axis=0) / 15]
    ranked_sentences = sorted(zip(averaged_cosine, sentences), key=lambda x:x[0])
    return ranked_sentences


def predict_annotated_queries(sents2ids, model_specifier=0, prompts_to_keep=None, prefix=None):
    """
    rank annotated queries only
    """
    model = None
    if model_specifier != 'random':
        model = init_ct_model(model_specifier)
    rankings = {}
    for prompt in tqdm(sents2ids):
        query = {}
        if 'Orsak:' in prompt:
            query['cause'] = prompt.split('Orsak: “')[1].split('”')[0]
        if 'Verkan:' in prompt:
            query['effect'] = prompt.split('Verkan: “')[1].split('”')[0]

        # print(query)
        if model is not None:
            rankings[prompt] = rank_selected_sentences(model, sents2ids[prompt],
                                                       query, prompts_to_keep)
        else:
            # random baseline
            rankings[prompt] = [[None, sentence] for sentence in sents2ids[prompt]]
            random.shuffle(rankings[prompt])
    outfile = f'{len(sents2ids)}_query_ranking_model_{model_specifier.replace("/", "")}.json'
    if prompts_to_keep:
        outfile = f"{'-'.join(prompts_to_keep)}_{outfile}"
    if prefix:
        outfile = f'{prefix}_{outfile}'
    with open(outfile, 'w') as ofile:
        json.dump(rankings, ofile, indent=4)


def fit_nn_model(embeddings: torch.Tensor,
                 n: int = 40) -> NearestNeighbors:
    start = time.time()
    nn = NearestNeighbors(n_neighbors=n, metric='cosine', p=1)
    nn.fit(embeddings)
    logging.debug(f'fit_nn_model took {time.time()-start} s')
    return nn


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
        contents = ([' '.join(text[n][1:3]),
                     text[n][3],
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


def run_ranking(prompts: List[str], embeddings: torch.Tensor,
                text: List[str],
                filter: Dict[str, int], n: int = 30,
                sorting_func=order_results_by_sents, emb_id: int = None,
                nn: NearestNeighbors = None):
    if len(prompts) > 1:
        # initial ranking with 'CAUSE medför EFFECT'
        # based on slightly higher reranking correlation
        i = 5
        prompt = prompts[i]
        reranking_prompts = torch.cat([prompts[:i],  prompts[i+1:]])
        prompts = torch.cat([torch.unsqueeze(prompt, dim=0),
                             reranking_prompts])
        top_k_dist, top_k_id = nn.kneighbors(
            torch.unsqueeze(embeddings[i], axis=0),
            n_neighbors=n)
        top_k_id = top_k_id[0]
        top_k_emb = torch.squeeze(torch.stack([embeddings[i]
                                               for i in top_k_id]))
        # rerank based on remaining prompts
        reranked_dist = torch.tensor(
            cosine_distances(torch.cat([embeddings[:i],
                                        embeddings[i+1:]]),
                             top_k_emb))
        dist = torch.cat([torch.tensor(top_k_dist), reranked_dist])
    else:
        print('looking at embeddings')
        # if len(embeddings.shape) <= 2:
        #    embeddings = torch.unsqueeze(embeddings[emb_id], dim=0)
        dist, top_k_id = nn.kneighbors(embeddings, n_neighbors=n)
        dist = torch.tensor(dist)
        top_k_id = top_k_id[0]
        print('sorting')
    return sorting_func(dist, top_k_id, prompts, text)


def generate_prompt_ranking(model, ids, query, embeddings, text, nn,
                            prompts_to_keep=None):
    prompts = generate_prompts(**query, restrict_to_prompt=prompts_to_keep)
    embedded_prompts = embed_text(**model, samples=prompts)
    return run_ranking(embedded_prompts, embeddings, text, {},
                       n=len(text),
                       nn=nn)


def generate_ranking(sents2ids: Dict, model_specifier: str,
                     embedding_path: str, text_path: str,
                     prompts_to_keep: List = None, prefix: str = None):
    """
    generate a full ranking of the entire data set. This is SLOW and not needed for the evaluation with the paired dataset!
    :param sents2ids: a dict of dicts over prompts mapping a tuple of sentences to their label, e.g. 
    sents2ids = {'Orsak: “droger”':
             {('', 'alkoholmissbruket orsakar problem.',
               ' Narkotikaproblemen är ett kriminal- och socialpolitiskt problem.'): 1,
              ('', 'Hantering av läkemedel medför kostnader.', ''): 2,
              ('', 'Ökad arbetsinsats vid källsortering leda till anpassningskostnader.', ''): 3},
             'Orsak: “klimatförändring”':
             {('', 'Krisveredskap kan indirekt påverkar miljön', ''): 1,
              ('', 'Jordbruket orsakar utsläpp av växthusgaser.', ''): 2,
              ('', 'Detta medför kostnader.', ''): 3,
              ('', 'Sammantaget bedöms klimatförändringens påverkan på skogsbilvägarna som stor och att det finns behov av ökad kunskap om hur man skall sköta och anpassa befintliga och tillkommande vägar till framtida förhållanden.', ''): 4}} 
    :param model_specifier: path to or name of the model to use for ranking ("random" for random baseline)
    :param embedding_path: path to the embedded corpus
    :param text_path: path to the sentences corresponding to the embeddings
    :param prompts_to_keep: if only a subset of the 13 keywords should be used for search specify which ones
    :param prefix: prefix for the output file
    """
    rankings = {}
    embeddings, text = load_documents(embedding_path, text_path)
    model = init_ct_model(model_specifier)
    nn = fit_nn_model(embeddings)
    for prompt in tqdm(sents2ids):
        query = {}
        if 'Orsak:' in prompt:
            query['cause'] = prompt.split('Orsak: “')[1].split('”')[0]
        if 'Verkan:' in prompt:
            query['effect'] = prompt.split('Verkan: “')[1].split('”')[0]

        # print(query)
        rankings[prompt] = generate_prompt_ranking(model, sents2ids[prompt],
                                                   query, embeddings, text, nn,
                                                   prompts_to_keep)

        #return generate_prompt_ranking(model, sents2ids[prompt],
        #                                           query, embeddings, text, nn)
    with open(f'{prefix}{len(sents2ids)}_full_prompt_ranking_model_{model_specifier.split("/")[-1]}.json',
              'w') as ofile:
        json.dump(rankings, ofile)


def plot_outliers(data_file, models):
    """
    embed example sentences and plot the avg over embedding dimensions
    """
    from matplotlib import pyplot as plt
    model = None
    plt.ylim([-12.5, 2.5])
    fig, axs = plt.subplots(len(models), 1, figsize=(10,15),
                            sharey=True, sharex=True,
                            constrained_layout=False)
    with open(data_file) as ifile:
        target_dict = json.load(ifile)
        targets = [sample['target sentence'] for sample in target_dict.values()]
    for i, model_name in enumerate(models):
        print(model_name)
        model_specifier = models[model_name]
        model = init_ct_model(model_specifier)

        # targets = targets[:5]
        rankings = {}
        embeddings = embed_text(**model, samples=targets)
        avg_embeddings = embeddings.mean(axis=0)
        print(embeddings.shape, avg_embeddings.shape)
        print('min:', set([tensor.item() for tensor in embeddings.argmin(axis=1)]),
              'max:', set([tensor.item() for tensor in embeddings.argmax(axis=1)]))
        print('min:', set([tensor.item() for tensor in embeddings.argmin(axis=0)]),
              'max:', set([tensor.item() for tensor in embeddings.argmax(axis=0)]))
        axs[i].plot(range(avg_embeddings.shape[0]), avg_embeddings)
        axs[i].set_title(model_name)
        axs[i].set_xlabel('embedding dimensions')
        axs[i].set_ylabel('vector average')
        for ax in axs.flat:
            ax.label_outer()
        fig.tight_layout(pad=3.0)
        fig.suptitle('Dimension average on 210 trial sentences', fontsize=16)


    # plt.plot(range(avg_embeddings.shape[0]), avg_embeddings)
    # print(f'{model_specifier.split("/")[-1]}_avg_embeddings.png')
    #fig.suptitle(f'{model_specifier.split("/")[-1]} dimension average on 210 trial sentences') 
    plt.savefig('combined_avg_embeddings.png')
    plt.clf()
    # dist = cosine_distances(embeddings)
    # print(dist)
    # print('avg. cosine distance:', sum(dist)/len(dist))
            

            
if __name__ == '__main__':
    text = read_html(
        '/Users/luisedu/Documents/Projekt/Causality/causality_code/evaluation_text.html')
    ids2sents, sents2ids = index_sentences(text)
    models = {'KB-BERT': "KB/bert-base-swedish-cased",
              'RISE-CT': "Contrastive-Tension/BERT-Base-Swe-CT-STSb",
              'CT-SOU only (1)': "/Users/luisedu/Documents/Projekt/Causality/causality_code/CT_BERT/STSbertb1",
              'CT-SOU only (2)': "/Users/luisedu/Documents/Projekt/Causality/causality_code/CT_BERT/STSbertb2",
              'RISE-CT + SOU (1)': "/Users/luisedu/Documents/Projekt/Causality/causality_code/CT_BERT/STSbertb1_sou_tuning",
              'RISE-CT + SOU (2)': "/Users/luisedu/Documents/Projekt/Causality/causality_code/CT_BERT/STSbertb2_sou_tuning"}
    # for model_name, path in models.items():  # needs all sentences embedded
    path = '/Users/luisedu/Documents/Projekt/Causality/'
    p = {}
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
    # predict_annotated_queries(sents2ids, models['KB-BERT'])
    for prompt in prompts:
        p[prompt] = predict_annotated_queries(sents2ids, models['RISE-CT'], [prompt])

  #plot_outliers('/Users/luisedu/Documents/Projekt/Causality/Swedish-Causality-Datasets/Binary-Trial-Data-Set/keyword_data_set.json',
    #              models)
    # To actually evaluate
    # predict_annotated_queries(sents2ids)
    # s2i = {k: sents2ids[k] for k in
    #        ['Orsak: “radon”',
    #         'Orsak: “luftföroreningar”, Verkan: “sjukdomar”',
    #         'Verkan: “sjukdomar”', 'Orsak: “luftföroreningar”']}
    # # p = generate_ranking(sents2ids, 0,
    # #                 f'{path}causality_demo/matches/match_embeddings.gzip',
    # #                 f'{path}causality_demo/matches/match_text.csv')
    # p = generate_ranking(sents2ids, f'{path}custom_ct',
    #                  f'{path}causality_demo/matches/match_embeddings.gzip',
    #                  f'{path}causality_demo/matches/match_text.csv')
