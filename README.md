# Spell Check

Based on the spell.py created by Peter Norvig, I have developed a spell class
and its children.

The basic idea in Norvig's speller is to use minimum edit distance to list 
word candidates (one or two edits away from the mistyped word) and 
select the most frequent word in a corpus as the best candidate.

I have reassemble the original code in a Spell class and used it to derived
different spell correctors. 

1. keyboardspell.py - also takes in account the distance between letter in a keyboard.
2. phoneticspell.py - uses eSpeak to the misspelled word phonetic transcription and 
looks for the word which has a similar phonetic transcription. It also consider 
minimum edit distance, but using elementary operations (addition, subtraction, transposition
and replacement) on phones. The distinction features description of phones is also
considered to measure phone dissimilarities. As the previous classes, frequency of
occurrence of words are also taken in account. An objective function is used to
balance between phone-edit distance and frequency of occurrence. 

