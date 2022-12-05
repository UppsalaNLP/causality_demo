import os
import spacy
path = os.path.realpath('__file__').strip('__file__')

# set up spacy model (downloaded from https://github.com/Kungbib/swedish-spacy)
model_path = f'{path}/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
if not os.path.exists(model_path):
    model_path = f'{path}/spacy_model/sv_model_xpos/sv_model0/sv_model0-0.0.0/'
model = spacy.load(model_path)
sentencizer = model.create_pipe('sentencizer')
model.add_pipe(sentencizer)



def redefine_boundaries(doc):
    """correct sentence boundaries of spacy sentencizer
    based on rules for abbreviation and possessive markers.
    returns the new sentences as strings and a list of lists
    of corresponding indeces on the document.
    Parameters:
               doc (spacy.tokens.doc.Doc):
                           a spacy Doc object representing the
                           annotated text
    """

    ents = [str(ent) for ent in doc.ents]
    sents = list(doc.sents)
    abr_exp = re.compile(r"(m\.m|osv|etc)\.")
    poss_exp = re.compile(r"\b[A-ZÄÖÅ0-9]+\b:$")
    token_sents = [[tok for tok in sent] for sent in sents]
    for i in range(len(sents)):
        if i+1 >= len(sents):
            break
        has_abbrev = abr_exp.findall(str(sents[i]))[::-1]
        if has_abbrev:
            if type(sents[i]) == Span:
                tokens = list(sents[i].__iter__())
            else:
                tokens = [token_sents[i]]#sents[i].split()
            last = None
            while has_abbrev:
                nb_abbr = len(has_abbrev)
                for j, t in enumerate(tokens):
                    if not has_abbrev:
                        break
                    if has_abbrev[-1] in str(t):
                        if j+1 < len(tokens) and\
                           (str(tokens[j+1]).istitle() and
                            str(tokens[j+1]) not in ents):
                            has_abbrev.pop(-1)
                            new_s = " ".join([str(tok) for tok in tokens[j+1:]])

                            following = sents[i+1:]
                            following_token_sents = token_sents[i+1:]
                            sents[i] = " ".join(
                                [str(tok) for tok in tokens[:j+1]])
                            token_sents[i] = [tok_i for tok_i in tokens[:j+1]]
                            token_sents[i+1] = [tok_i for tok_i in tokens[j+1:]]
                            token_sents = token_sents[:i+2]
                            token_sents.extend(following_token_sents)
                            sents[i+1] = new_s
                            sents = sents[:i+2]
                            sents.extend(following)
                if nb_abbr == len(has_abbrev):
                    has_abbrev.pop(-1)

        # possessives of acronyms etc. tend to get split at the colon
        # i.e. 'EU:s direktiv ...' -> 'EU:', 's direktiv ...'
        has_poss = poss_exp.findall(str(sents[i]))
        split_on_poss = (has_poss and
                         (i + 1 < len(sents)
                          and re.match('[a-zäåö]', str(sents[i+1])[:2])))
        if split_on_poss:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1",
                              str(sents[i]) + str(sents[i+1]))
            del sents[i+1]
            token_sents[i].extend(token_sents[i+1])
            del token_sents[i+1]
        else:
            sents[i] = re.sub(r" ([.,;:!?])", r"\1", str(sents[i]))
        # sentences that start with parentheses are split at open parentheses
        if str(sents[i]).endswith("(") and i + 1 < len(sents):
            len_before = len(sents[i])
            removed = list(sents[i])
            sents[i] = str(sents[i]).rstrip(' (')
            removed = list(filter(lambda x: x != ' ',
                             removed[-(len_before - len(sents[i])):]))
            if removed:
                additions = token_sents[i][-1*len(removed):]
                token_sents[i] = token_sents[i][:-1*len(removed)]
                token_sents[i+1] = additions + token_sents[i+1]
            sents[i+1] = '(' + str(sents[i+1]).lstrip()
        sents = [str(s) for s in sents]
    return sents, token_sents
