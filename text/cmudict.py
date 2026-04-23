""" from https://github.com/keithito/tacotron """

import re


valid_symbols = [

  'pʲː',
  'ə',
  'cː',
  'tsʲ',
  'ɨ',
  'ʎː',
  'ʂː',
  'u',
  'ɟː',
  'ʐː',
  'ʉ',
  'ʂ',
  'z̪ː',
  'dʐː',
  't̪s̪',
  'mʲː',
  'ʑː',
  'bː',
  'fː',
  'b',
  't̪',
  'ɫː',
  'ɡː',
  'ɲː',
  'sʲː',
  'jː',
  'ɡ',
  'ɟ',
  'ɐ',
  'j',
  'mː',
  's',
  'sʲ',
  'ɪ',
  'm',
  'x',
  'd̪',
  'z̪ː',
  'd̪z̪ː',
  'tɕː',
  'tʲː',
  'rʲ',
  'fʲː',
  'c',
  'xː',
  'v',
  'rʲː',
  'pʲ',
  'i',
  'ʎ',
  'bʲ',
  'tɕ',
  'tʂ',
  'vʲ',
  't̪s̪',
  'ʐ',
  'o',
  'vː',
  'kː',
  'd̪z̪',
  't̪ː',
  'rː',
  'ɫ',
  'e',
  'ɣ',
  'tʲ',
  'dzʲː',
  'tʂː',
  'p',
  'zʲː',
  's̪',
  'ɵ',
  'z̪',
  'r',
  'ɲ',
  'bʲː',
  'ɕː',
  'dʲː',
  'ɕ',
  'pː',
  'ɛ',
  'zʲ',
  'mʲ',
  'v',
  's̪ː',
  'vʲː',
  'n̪ː',
  'spn',
  't̪s̪ː',
  'dʲ',
  'ʊ',
  'fʲ',
  'æ',
  'd̪',
  'd̪ː',
  'k',
  'n̪',
  'ç',
  'a',
  'f',

  # M2: stressed vowel variants (MFA stressed-position labels)
  'a+',
  'e+',
  'i+',
  'o+',
  'u+',
  'ɨ+',

]

_valid_symbol_set = set(valid_symbols)


class CMUDict:
    """Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict"""

    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding="latin-1") as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of ARPAbet pronunciations of the given word."""
        return self._entries.get(word.upper())


_alt_re = re.compile(r"\([0-9]+\)")


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= "А" and line[0] <= "Я" or line[0] >= "а" and line[0] <= "я" or line[0] == "'"):
            parts = line.split("  ")
            word = re.sub(_alt_re, "", parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(" ")
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return " ".join(parts)
