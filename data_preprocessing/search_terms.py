import requests
import pickle
import re
import os
search_terms = ['"bero på"',
                '"bidra till"',
                '"leda till"',
                '"på grund av"',
                '"till följd av"',
                '"är ett resultat av"',
                "resultera",
                "förorsaka",
                "orsaka",
                "påverka",
                "effekt",
                "medföra",
                "framkalla",
                "vålla"]

annotated_search_terms = [('"bero på"', 0, "vb"),
                ('"bidra till"', 0, "vb"),
                ('"leda till"', 0, "vb"),
                ('"på grund av"', 1, "nn"),
                ('"till följd av"', 1, "nn"),
                ('"vara ett resultat av"', 0, "vb"),
                ("resultera", 0, "vb"),
                ("förorsaka", 0, "vb"),
                ("orsaka", 0, "vb"),
                ("påverka", 0, "vb"),
                ("medföra",0, "vb"),
                ("framkalla", 0, "vb"),
                ("vålla", 0, "vb")]


def expand(term_list=annotated_search_terms, dictionary=None):
    if dictionary is None:
        dictionary = {"vara": ["var", "är", "varit", "vore", "vara"]}
    terms = {}
    for term, i, pos in term_list:
        if term not in terms:
            terms[term] = set()
        if pos:
            words = term.strip("\"").split()
            if words[i] in dictionary:
                if len(words) > 1:
                    for form in dictionary[words[i]]:
                        if i + 1 == len(words):
                            terms[term].add(f'"{" ".join(words[:i] + [form])}"')
                        else:
                            terms[term].add(f'"{" ".join(words[:i] + [form] + words[min(i+1, len(words)-1):])}"')
                else:
                    terms[term] = terms[term].union(dictionary[words[i]])
            else:
                forms = requests.get(f"https://skrutten.csc.kth.se/granskaapi/inflect.php?word={words[i]}&tag={pos}")
                if forms:
                    wforms = []
                    for form in forms.text.split('<br>'):
                        form = form.strip()
                        if f'&lt;{pos}' in form and not form.startswith("all forms"):
                            wforms.append(form.split('&lt;')[0].strip())
                        # else:
                        #    print('no:', form)
                    #forms = re.sub("((&lt;)+[a-z.*]*(&gt;)+|<br>|\d+)", "", forms.text.strip())
                    #forms = [el.split()[0].strip() for el in forms.split("\n")[1:]
                             #if el.split() and not el.startswith("all forms")]
                    dictionary[words[i]] = forms
                    for form in set(wforms):
                        if len(words) > 1:
                            terms[term].add(f'"{" ".join(words[:i] + [form] + words[min(i+1, len(words)-1):])}"')
                        else:
                            terms[term].add(form)
        else:
            terms[term].add(term)
    return terms

def create_tagged_term_list(term_dict, term_annotations):
    """
    returns a list of all terms in term dict and their pos information
    in the style of the parsed_target field in the index schema
    """
    tagged_list = []
    for term, _, pos in term_annotations:
        if '"' in term:
            tagged_list.extend(term_dict[term])
        else:
            tagged_list.extend([f'{form}//{pos}' for form in term_dict[term]])
    return tagged_list

expanded_dict = {'"bero på"': {'"berodd på"', '"beror på"', '"berodds på"', '"beros på"', '"berodda på"',
                               '"berott på"', '"berotts på"', '"beroddes på"', '"berodde på"', '"bero på"'},
                 '"bidra till"': {'"bidragna till"', '"bidragits till"', '"bidrog till"', '"bidra till"',
                                  '"bidragande till"', '"bidragit till"', '"bidrogs till"', '"bidras till"',
                                  '"bidrar till"', '"bidragne till"'},
                 '"leda till"': {'"ledda till"', '"led till"', '"ledar till"', '"leder till"', '"lett till"',
                                 '"ledades till"', '"ledds till"', '"ledes till"', '"leda till"', '"ledde till"',
                                 '"ledad till"', '"ledats till"', '"ledd till"', '"ledas till"', '"ledat till"',
                                 '"ledads till"', '"ledande till"', '"letts till"', '"ledade till"', '"leddes till"'},
                 '"på grund av"': {'"på grunders av"', '"på grundet av"', '"på grunder av"', '"på grunds av"',
                                   '"på grundets av"', '"på grunden av"', '"på grund av"', '"på grundens av"',
                                   '"på grunderna av"', '"på grundernas av"'},
                 '"till följd av"': {'"till följds av"', '"till följden av"', '"till följd av"', '"till följdens av"',
                                     '"till följder av"', '"till följdernas av"', '"till följderna av"', '"till följders av"'},
                 '"vara ett resultat av"': {'"vara ett resultat av"', '"är ett resultat av"', '"var ett resultat av"',
                                            '"vore ett resultat av"', '"varit ett resultat av"'},
                 'resultera': {'resulterats', 'resulterat', 'resulterar', 'resulterads', 'resulterade', 'resulterande',
                               'resulterad', 'resulteras', 'resultera', 'resulterades'},
                 'förorsaka': {'förorsakads', 'förorsakas', 'förorsakar', 'förorsakande', 'förorsakades', 'förorsakade',
                               'förorsakats', 'förorsakat', 'förorsakad', 'förorsaka'},
                 'orsaka': {'orsakades', 'orsakad', 'orsaka', 'orsakads', 'orsakande', 'orsakade', 'orsakat', 'orsakats',
                            'orsakar', 'orsakas'},
                 'påverka': {'påverkats', 'påverkads', 'påverkas', 'påverkade', 'påverkar', 'påverkad', 'påverkades',
                             'påverkande', 'påverka', 'påverkat'},
                 'medföra': {'medförts', 'medförd', 'medföra', 'medföras', 'medförda', 'medföres', 'medförds', 'medförde',
                             'medför', 'medfördes', 'medfört'},
                 'framkalla': {'framkallas', 'framkallade', 'framkallads', 'framkallades', 'framkallats', 'framkallat',
                               'framkallad', 'framkalla', 'framkallar', 'framkallande'},
                 'vålla': {'vållats', 'vållad', 'vållat', 'vållades', 'vållas', 'vållande', 'vållar', 'vållade',
                           'vålla', 'vållads'}}

# restricted based on pos (maybe it is wise to keep the old list for now)
new_expanded_dict = {'"bero på"': {'"beror på"', '"berodde på"', '"bero på"',
                                   '"berott på"', '"beros på"', '"beroddes på"',
                                   '"berotts på"'},
                     '"bidra till"': {'"bidrar till"', '"bidras till"', '"bidrog till"',
                                      '"bidra till"', '"bidragits till"',
                                      '"bidrogs till"', '"bidragit till"'},
                     '"leda till"': {'"leder till"', '"ledar till"', '"ledde till"',
                                     '"ledades till"', '"lett till"', '"ledes till"',
                                     '"ledas till"', '"leda till"', '"letts till"',
                                     '"ledade till"', '"ledats till"', '"ledat till"',
                                     '"leddes till"', '"led till"'},
                     '"på grund av"': {'"på grunderna av"', '"på grunder av"',
                                       '"på grundernas av"', '"på grundets av"',
                                       '"på grund av"', '"på grunds av"', '"på grunders av"',
                                       '"på grunden av"', '"på grundens av"', '"på grundet av"'},
                     '"till följd av"': {'"till följd av"', '"till följders av"', '"till följder av"',
                                         '"till följds av"', '"till följden av"', '"till följdernas av"',
                                         '"till följdens av"', '"till följderna av"'},
                     '"vara ett resultat av"': {'"vore ett resultat av"', '"varit ett resultat av"',
                                                '"är ett resultat av"', '"vara ett resultat av"',
                                                '"var ett resultat av"'},
                     'resultera': {'resulterats', 'resultera', 'resulteras',
                                   'resulterat', 'resulterade', 'resulterar', 'resulterades'},
                     'förorsaka': {'förorsakade', 'förorsakas', 'förorsakar', 'förorsaka',
                                   'förorsakat', 'förorsakades', 'förorsakats'},
                     'orsaka': {'orsakas', 'orsakat', 'orsakar', 'orsakades',
                                'orsakade', 'orsaka', 'orsakats'},
                     'påverka': {'påverkades', 'påverkats', 'påverkas', 'påverka',
                                 'påverkat', 'påverkade', 'påverkar'},
                     'medföra': {'medfört', 'medför', 'medföras', 'medförde',
                                 'medförts', 'medföres', 'medföra', 'medfördes'},
                     'framkalla': {'framkallas', 'framkalla', 'framkallat',
                                   'framkallades', 'framkallar', 'framkallade', 'framkallats'},
                     'vålla': {'vållade', 'vållades', 'vållats', 'vålla', 'vållar', 'vållat', 'vållas'}}

# filter out unlikely phrases
# removed: '"på grunderna av"', '"på grundernas av"',
# '"på grundets av"', '"på grunders av"',  '"på grundens av"',
# '"på grundet av"', '"till följders av"', '"till följdens av"',
# '"till följdernas av"', '"till följds av"',

# to add:
# var resultatet av, var resultaten av
filtered_expanded_dict = {'"bero på"': {'"beror på"', '"berodde på"', '"bero på"',
                                   '"berott på"', '"beros på"', '"beroddes på"',
                                   '"berotts på"'},
                     '"bidra till"': {'"bidrar till"', '"bidras till"', '"bidrog till"',
                                      '"bidra till"', '"bidragits till"',
                                      '"bidrogs till"', '"bidragit till"'},
                     '"leda till"': {'"leder till"', '"ledar till"', '"ledde till"',
                                     '"ledades till"', '"lett till"', '"ledes till"',
                                     '"ledas till"', '"leda till"', '"letts till"',
                                     '"ledade till"', '"ledats till"', '"ledat till"',
                                     '"leddes till"', '"led till"'},
                     '"på grund av"': {'"på grunder av"', '"på grund av"', '"på grunds av"',
                                       '"på grunden av"'},
                     '"till följd av"': {'"till följd av"', '"till följder av"',
                                         '"till följden av"', '"till följderna av"'},
                     '"vara ett resultat av"': {'"vore ett resultat av"', '"varit ett resultat av"',
                                                '"är ett resultat av"', '"vara ett resultat av"',
                                                '"var ett resultat av"'},
                     'resultera': {'resulterats', 'resultera', 'resulteras',
                                   'resulterat', 'resulterade', 'resulterar', 'resulterades'},
                     'förorsaka': {'förorsakade', 'förorsakas', 'förorsakar', 'förorsaka',
                                   'förorsakat', 'förorsakades', 'förorsakats'},
                     'orsaka': {'orsakas', 'orsakat', 'orsakar', 'orsakades',
                                'orsakade', 'orsaka', 'orsakats'},
                     'påverka': {'påverkades', 'påverkats', 'påverkas', 'påverka',
                                 'påverkat', 'påverkade', 'påverkar'},
                     'medföra': {'medfört', 'medför', 'medföras', 'medförde',
                                 'medförts', 'medföres', 'medföra', 'medfördes'},
                     'framkalla': {'framkallas', 'framkalla', 'framkallat',
                                   'framkallades', 'framkallar', 'framkallade', 'framkallats'},
                     'vålla': {'vållade', 'vållades', 'vållats', 'vålla', 'vållar', 'vållat', 'vållas'}}

# öka tillta, minska avta växa?, ökning tillväxt höjning, minskning nedgång reducering avtagande
# wonder if there is a preference for POS

increase_terms = ['öka', 'tillta',  'växa', 'ökning', 'uppgång', 'tilltagande', 'höjning']
annotated_increase_terms = [('öka', 0, 'vb'),
                            ('tillta', 0, 'vb'),
                            ('växa', 0, 'vb'),
                            ('ökning', 0, 'nn'),
                            ('uppgång', 0, 'nn'),
                            ('tilltagande', 0, 'nn'),
                            ('höjning', 0, 'nn')]
incr_dict = {'öka': {'ökades', 'ökats', 'ökar', 'ökade', 'öka', 'ökat', 'ökas'},
             'tillta': {'tilltogs', 'tilltagit', 'tillta', 'tilltas', 'tilltog', 'tilltagits', 'tilltar'},
             'växa': {'växa', 'växs', 'växer', 'växte', 'vuxit', 'väx', 'vuxits', 'växtes', 'växas', 'växts', 'växt'},
             'ökning': {'ökningarna', 'ökningen', 'ökningens', 'ökningars', 'ökning', 'ökningarnas', 'ökningar'},
             'uppgång': {'uppgångarnas', 'uppgången', 'uppgångarna', 'uppgångens', 'uppgångar', 'uppgångars', 'uppgång'},
             'tilltagande': {'tilltagande', 'tilltagandet', 'tilltagandes',  'tilltagandets'},
             'höjning': {'höjningar', 'höjningens', 'höjningars', 'höjningarna', 'höjning', 'höjningen', 'höjningarnas'}}

decrease_terms = ['minska', 'avta','minskning', 'nedgång', 'avtagande', 'sänkning']
annotated_decrease_terms = [('minska', 0, 'vb'),
                            ('avta', 0, 'vb'),
                            ('minskning', 0, 'nn'),
                            ('nedgång', 0, 'nn'),
                            ('avtagande', 0, 'nn'),
                            ('sänkning', 0, 'nn')]
decr_dict = {'minska': {'minskade', 'minskar', 'minska', 'minskas', 'minskat', 'minskats', 'minskades'},
             'avta': {'avtogs', 'avtas', 'avtar', 'avtog', 'avta', 'avtagits', 'avtagit'},
             'minskning': {'minskningar', 'minskningars', 'minskningen', 'minskning', 'minskningarna', 'minskningens', 'minskningarnas'},
             'nedgång': {'nedgången', 'nedgångarnas', 'nedgångarna', 'nedgångars', 'nedgångens', 'nedgångar', 'nedgång'},
             'avtagande':  {'avtagande', 'avtagandet', 'avtagandes', 'avtagandets'},
             'sänkning': {'sänkning', 'sänkningars', 'sänkningarnas', 'sänkningarna', 'sänkningen', 'sänkningar', 'sänkningens'}}


keys_to_pos = {'minska': 'VB',
                 'avta': 'VB',
                 'minskning': 'NN',
                 'nedgång': 'NN',
                 'avtagande': 'NN',
                 'sänkning': 'NN',
                 'öka': 'VB',
                 'tillta': 'VB',
                 'växa': 'VB',
                 'ökning': 'NN',
                 'uppgång': 'NN',
                 'tilltagande': 'NN',
                 'höjning': 'NN'}

tagged_list = ['"berodde på"',
               '"beroddes på"',
               '"bero på"',
               '"beror på"',
               '"beros på"',
               '"berott på"',
               '"berotts på"',
               '"bidrar till"',
               '"bidragits till"',
               '"bidras till"',
               '"bidrogs till"',
               '"bidra till"',
               '"bidrog till"',
               '"bidragit till"',
               '"led till"',
               '"ledat till"',
               '"ledas till"',
               '"ledats till"',
               '"ledade till"',
               '"ledde till"',
               '"letts till"',
               '"ledes till"',
               '"ledar till"',
               '"leddes till"',
               '"ledades till"',
               '"lett till"',
               '"leda till"',
               '"leder till"',
               '"på grund av"',
               '"på grunden av"',
               '"på grunds av"',
               '"på grunder av"',
               '"till följd av"',
               '"till följderna av"',
               '"till följden av"',
               '"till följder av"',
               '"vore ett resultat av"',
               '"var ett resultat av"',
               '"varit ett resultat av"',
               '"vara ett resultat av"',
               '"är ett resultat av"',
               'resulterade//vb',
               'resulteras',
               'resulterat//vb',
               'resulterats',
               'resulterades',
               'resulterar',
               'resultera',
               'förorsakas',
               'förorsakats',
               'förorsakat//vb',
               'förorsaka',
               'förorsakar',
               'förorsakade//vb',
               'förorsakades',
               'orsakas',
               'orsakade//vb',
               'orsakat//vb',
               'orsakades',
               'orsaka',
               'orsakats',
               'orsakar',
               'påverka',
               'påverkat//vb',
               'påverkats',
               'påverkades',
               'påverkade//vb',
               'påverkar',
               'påverkas',
               'medför',
               'medföra',
               'medföres',
               'medfördes',
               'medföras',
               'medförde//vb',
               'medförts',
               'medfört//vb',
               'framkallas',
               'framkallade//vb',
               'framkalla',
               'framkallades',
               'framkallar',
               'framkallat//vb',
               'framkallats',
               'vållar',
               'vållades',
               'vållade//vb',
               'vållas',
               'vålla',
               'vållats',
               'vållat//vb']
