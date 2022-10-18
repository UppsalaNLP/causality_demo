import glob
import os
import csv
from tqdm import tqdm
import re
import sys
sys.path.append('/Users/luisedu/Documents/Projekt/Causality/old_demo/causality_demo/data_preprocessing')
# sys.path.append('/Users/luidu652/Documents/causality_extraction/')
from segmentation import redefine_boundaries, model
from data_extraction import Text
from search_terms import annotated_search_terms, filtered_expanded_dict,\
    create_tagged_term_list


def scan_files(filenames):
    sents = 0
    counter = 0
    tagged_list = create_tagged_term_list(filtered_expanded_dict,
                                          annotated_search_terms)

    with open('nonmatches.csv', 'w') as ofile:
        writer = csv.writer(ofile, delimiter=';')
        for i, name in enumerate(filenames):
            counter += 1
            if i % 500 == 0:
                print(f'at file nb {i}: {name}')
            for id_, n, match in scan_text(name, tagged_list):
                sents += n
                writer.writerow([name, id_, match])
    print(counter, sents)


def tag_file(filename):
    doc_name = filename.split('/')[-1]
    with open(f'tagged_docs/tagged_{doc_name}', 'w') as ofile:
        writer = csv.writer(ofile, delimiter=';')
        text = Text('')
        text.from_html(filename)
        for section in text:
            for paragraph in section:
                doc = model(paragraph)
                sentences, token_s = redefine_boundaries(doc)
                for i, sentence in enumerate(token_s):
                    writer.writerow([section.title,
                                     ' '.join(
                                         ['//'.join([tok.text, tok.tag_,
                                                    tok.dep_, str(tok.head.i)])
                                                    for tok in sentence])])
            writer.writerow([])


def segment_file(filename):
    doc_name = filename.split('/')[-1]
    with open(f'sou_docs/{doc_name.split(".")[0]}', 'w') as ofile:
        writer = csv.writer(ofile, delimiter=';')
        text = Text('')
        text.from_html(filename)
        for section in text:
            for paragraph in section:
                doc = model(paragraph)
                sentences, token_s = redefine_boundaries(doc)
                for i, sentence in enumerate(token_s):
                    writer.writerow([section.title,
                                     re.sub(r'([([{]) ', r'\1',
                                            re.sub(r' ([,;.:!?})\]])',
                                                   r'\1',                                 
                                                   ' '.join(
                                                       [tok.text for tok
                                                        in sentence])))])
            writer.writerow([])


def scan_text(file, keywords):
    text = Text('')
    sent_count = 0
    text.from_html(file)
    for section in text:
        for paragraph in section:
            doc = model(paragraph)
            sentences, token_s = redefine_boundaries(doc)
            for i, sentence in enumerate(sentences):
                sent_count += 1
                if not matches(str(sentence).split(), token_s[i], keywords):
                    yield i, sent_count, sentence
                    sent_count = 0


def matches(sentence, token_s, keywords):
    for keyword in keywords:
        if '//' in keyword:
            token, tag = keyword.split('//')
            for tok in token_s:
                if tok.tag_.startswith(tag) and\
                   tok.text == token:
                    return True
        elif keyword.startswith('"'):
            part_match = 0
            phrase = keyword.strip('"').split()
            for tok in sentence:
                if part_match == len(phrase):
                    return True
                # slop ?
                if tok == phrase[part_match]:
                    part_match += 1
                elif tok == phrase[max(part_match, 0)]:
                    pass
        elif keyword in sentence:
            return True


def remove_duplicates(file):
    with open('duplicates') as ifile:
        duplicates = [line.split()[-1] for line in ifile]
    previously_seen = {}
    token = r"(([^ ;]+|;)//([-A-Z|]+(/[-A-Z|])?)*//[a-zA-Z:]+//\d+)"
    exp = re.compile(r"(tagged_docs/tagged_ft_[A-Za-z0-9]+\.csv:\d+:[^;]+|" +
                     rf"({token}( {token})*)|''|\"\")")
    with open(file) as ifile,\
         open('non_matches_unique.csv', 'w') as ofile:
        # reader = csv.reader(ifile, delimiter=';', quotechar="'", doublequote=True)
        writer = csv.writer(ofile, delimiter=';')
        for line in tqdm(ifile):
            # print(len(previously_seen), len(duplicates))
            fields = exp.findall(line)
            line = [el[0].strip('"').strip("'") for el in fields]
            if len(line) > 4:
                line[3] = " ".join(line[3:])
                line = line[:4]
            if len(line) != 4:
                print(line, len(line))
                continue
            if line[0] in duplicates:
                if line[0] in previously_seen:
                    other_line = previously_seen[line[0]]
                    if not other_line[-1] and line[2] and line[-1]:
                        writer.writerow(line)
                    else:
                        writer.writerow(line)
            else:
                writer.writerow(line)


def fix_format(file):
    path = file.split('/')
    filename = path[-1]
    path = '/'.join(path[:-1])
    if path:
        path += '/'
    with open(file) as ifile,\
         open(f'{path}untagged_{filename}', 'w') as ofile,\
         open(f'{path}tagged_{filename}', 'w') as tagged_ofile:
        writer = csv.writer(ofile, delimiter=';', quoting=csv.QUOTE_ALL)
        tagged_writer = csv.writer(tagged_ofile, delimiter=';', quoting=csv.QUOTE_ALL)

        current_example = []
        previous = ''
        position_counter = 0
        is_match = False
        for line in ifile:
            position_counter += 1
            line = line.rstrip().split(';', 1)
            if len(line) < 2:
                line.append('')
            # end of segment
            if line[0] == '--' and current_example:
                current_example.append(previous)
                assert len(current_example) == 4, f'{len(current_example)} {current_example}'
                tagged_writer.writerow(current_example)
                current_example = [current_example[0]] + [
                    re.sub(r'([([{]) ', r'\1',
                           re.sub(r' ([,;.:!?})\]])', r'\1',
                                  ' '.join([el.split('//')[0] for el in line.split()])))
                    for line in current_example[1:]]
                writer.writerow(current_example)
                is_match = False
                position_counter = 0
                current_example = None
                continue
            elif is_match:
                current_example.append(line[1])
                assert len(current_example) == 4, f'{len(current_example)} {current_example}'
                tagged_writer.writerow(current_example)
                current_example = [current_example[0]] + [
                    re.sub(r'([([{]) ', r'\1',
                           re.sub(r' ([,;.:!?})\]])', r'\1',
                                  ' '.join([el.split('//')[0] for el in line.split()])))
                    for line in current_example[1:]]
                writer.writerow(current_example)
                is_match = False
                current_example = None
            # match
            if 'csv:' in line[0]:
                document = line[0]
                current_example = [document, previous, line[1]]
                is_match = True
            # context
            previous = line[1]


def fix_format_2(file, n=2):
    """
    fix format but with n context sentences on each side
    """
    path = file.split('/')
    filename = path[-1]
    path = '/'.join(path[:-1])
    if path:
        path += '/'
    line_nb_exp = re.compile(r'csv[-:](\d+)[-:]\d+')
    with open(file) as ifile,\
         open(f'{path}untagged_{filename}', 'w') as ofile,\
         open(f'{path}tagged_{filename}', 'w') as tagged_ofile:
        writer = csv.writer(ofile, delimiter=';', quoting=csv.QUOTE_ALL)
        tagged_writer = csv.writer(tagged_ofile, delimiter=';', quoting=csv.QUOTE_ALL)

        current_segment = []
        match_ids = []
        match_docs = []
        for line in ifile:
            line = line.rstrip().split(';', 1)
            section = re.split(r'csv[-:]\d+[-:]\d+', line[0])[-1]
            line_nb = line_nb_exp.findall(line[0])
            if line_nb:
                line_nb = int(line_nb[0])
            if len(line) < 2:
                line.append('')
            # end of segment
            if line[0] == '--':
                write_matches(match_ids, current_segment, match_docs,
                              n, tagged_writer, writer)
                match_docs = []
                match_ids = []
                current_segment = []
                continue
            # match
            if 'csv:' in line[0]:
                match_docs.append(line[0])
                match_ids.append(len(current_segment))
            # context
            current_segment.append(line[1])
        write_matches(match_ids, current_segment, match_docs,
                      n, tagged_writer, writer)


def write_matches(match_ids, current_segment, match_docs,
                  n, tagged_writer, writer):
    # generate context for each match
    for i in match_ids:
        start = max(i - n, 0)
        stop = min(i + n + 1, len(current_segment))
        slice = current_segment[start:stop]
        # missing left context
        left_difference = i - start
        if left_difference < n:
            slice = [''] * (n - left_difference) + slice
        # missing right context
        right_difference = stop - (i + 1)
        if right_difference < n:
            slice = slice + [''] * (n - right_difference)
        assert len(slice) == 1 + n*2,\
            f'example length should be {1 + n*2}, but is ' +\
            f'{len(slice)} ({start, i, stop}) ({len(current_segment)}) for {slice}'
        match_doc = match_docs.pop(0)
        tagged_writer.writerow([match_doc] + slice)
        untagged_example = [match_doc] + [
            re.sub(r'([([{]) ', r'\1',
                   re.sub(r' ([,;.:!?})\]])', r'\1',
                          ' '.join([el.split('//')[0] for el in sent.split()])))
            for sent in slice]
        writer.writerow(untagged_example)


def untag_docs():
    files = glob.glob('tagged_docs/*.html')
    print(len(files), files[0])
    out_dir = 'sou_docs'
    if not os.path.exists(out_dir):
        os.system(f'mkdir {out_dir}')
    for file in tqdm(files):
        doc_id = file.split('/tagged_')[-1].strip('.html')
        with open(file) as ifile, open(f'{out_dir}/{doc_id}', 'w') as ofile:
            writer = csv.writer(ofile, delimiter=';', quoting=csv.QUOTE_ALL)
            for line in ifile:
                line = line.rstrip().split(';', 1)
                line[-1] = re.sub(r'([([{]) ', r'\1',
                                  re.sub(r' ([,;.:!?})\]])', r'\1',
                                         ' '.join([el.split('//')[0]
                                                   for el in line[-1].split()])))
                writer.writerow(line)


def collect_all_sentences(prefix='ft'):
    files = glob.glob(f'sou_docs/{prefix}*')
    with open('corpus.txt', 'w') as ofile:
        for file in files:
            with open(file) as ifile:
                reader = csv.reader(ifile, delimiter=';')
                for line in reader:
                    print(line[-1], file=ofile)


if __name__ == '__main__':
    if input('run tagging? (y/n)').casefold() == 'y':
        files = glob.glob('documents/s_*.html')
        files += glob.glob('documents/SEs_*.html')
        for file in tqdm(files):
            tag_file(file)
    elif input('remove duplicates? (y/n)').casefold() == 'y':
        remove_duplicates('unique_segmented_tagged.nm.csv')
    elif input('retrieve causality keyword matches? (y/n)') == 'y':
        os.system('./extraction.sh')
        fix_format_2('matches_cw_2.csv')
    elif input('segment files? (y/n)').casefold() == 'y':
        files = glob.glob('documents/s_*.html')
        for file in tqdm(files):
            segment_file(file)
