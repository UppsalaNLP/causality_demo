# Data preprocessing

## Setup
   For tagging and segmentation we used https://github.com/Kungbib/swedish-spacy (XPOS).
   Download it and update the path in segmentation.py.
   **Edit:** you can now download it with pip and directly load it.

## 1) Restructure the html files on https://data.riksdagen.se using extract_from_html (data_extraction)
   Follow run_example. This reads in document metadata from documents.pickle (title, date and link to the html of the report), parses the corresponding html and finally creates the files in https://github.com/UppsalaNLP/SOU-corpus/tree/master/html.

## 2) Tagging
   Use tag_file in corpus_processing to POS-tag the sentences. Output should look like https://github.com/UppsalaNLP/SOU-corpus/tree/master/tagged. This also adds the section header as first element in the output csv!

## 3) Filtering
   We filter with extraction.sh. Update input and output paths as necessary.

## 4) Post-processing
   Inspect the files! I remember that in directly adjacent matches, the context window can be messed up.
   Use fix_format_2 in corpus_processing.py can be used to fix that and to generate both a tagged and untagged version of all matches in context. This should generate files in the format of the match_text.csv.

## 5) Sentence embeddings
   Use the sentence-embedding model to generate representations for the matched sentence (i.e. the one in the centre). I seem to have lost my original code for that, but it is as simple running embed_text (causality_demo) on a bigger batch and saving the concatenated tensor.