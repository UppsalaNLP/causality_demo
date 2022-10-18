#!/usr/bin/env bash
# to match
# query with line nbs
ggrep -n -2 -P '\b(bero(ddes?|tts?|s|r)?//VB[^ ]+( \w+[^ ]+)? på//PP[^ ]+ |(bidra(r|s|gits?)?|bidrogs?|led(a([rs]|des?|ts?)?|d?es?|er)?|letts?)//VB[^ ]+( \w+[^ ]+)? till//PP|på//PP[^ ]+( \w+[^ ]+)? grund(e[rn]|s)?//NN[^ ]+( \w+)? av//PP[^ ]+ |(var(a|it)?|vore|är)//VB[^ ]+( \w+[^ ]+)? ett//DT[^ ]+( \w+[^ ]+)? resultat//NN[^ ]+( \w+[^ ]+)? av//PP[^ ]+|till//PP[^ ]+( \w+[^ ]+)? följd(e[nr]|erna)?//NN[^ ]+( \w+[^ ]+)? av//PP|(påverka|resultera|(för)?orsaka|framkalla|vålla)(r|s|ts?|des?)?//VB|medför(as?|es|des?|ts?)?//VB)' corpus/tagged/* > matches_cw_2.csv

# anything but the query
# ggrep -n -1 -vP '\b(bero(ddes?|tts?|s|r)?//VB[^ ]+( \w+[^ ]+)? på//PP[^ ]+ |(bidra(r|s|gits?)?|bidrogs?|led(a([rs]|des?|ts?)?|d?es?|er)?|letts?)//VB[^ ]+( \w+[^ ]+)? till//PP|på//PP[^ ]+( \w+[^ ]+)? grund(e[rn]|s)?//NN[^ ]+( \w+)? av//PP[^ ]+ |(var(a|it)?|vore|är)//VB[^ ]+( \w+[^ ]+)? ett//DT[^ ]+( \w+[^ ]+)? resultat//NN[^ ]+( \w+[^ ]+)? av//PP[^ ]+|till//PP[^ ]+( \w+[^ ]+)? följd(e[nr]|erna)?//NN[^ ]+( \w+[^ ]+)? av//PP|(påverka|resultera|(för)?orsaka|framkalla|vålla)(r|s|ts?|des?)?//VB|medför(as?|es|des?|ts?)?//VB)' tagged_docs/* > parsed_non_matches.csv
