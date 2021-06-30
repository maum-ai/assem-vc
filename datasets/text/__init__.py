""" from https://github.com/keithito/tacotron """
import re
from . import cleaners
from .symbols import eng_symbols, cmu_symbols, kor_symbols, cht_symbols


class Language():
  def __init__(self, lang, text_cleaners, use_eos=True):
    # Regular expression matching text enclosed in curly braces:
    self._curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

    self._use_eos = use_eos
    if lang == 'kor':
      available_cleaners = ['korean_cleaners']
      symbols = kor_symbols
    elif lang in ('eng', 'eng2'):
      available_cleaners = ['basic_cleaners',
                            'transliteration_cleaners',
                            'english_cleaners']
      symbols = eng_symbols
    elif lang == 'cmu':
      available_cleaners = ['basic_cleaners',
                            'transliteration_cleaners',
                            'english_cleaners']
      symbols = cmu_symbols
    elif lang == 'cht':
      available_cleaners = ['chinese_cleaners']
      symbols = cht_symbols
    elif lang == 'jap_romaji' or lang == 'jap':
      available_cleaners = ['japanese_romaji_cleaners', 'japanese_cleaners']
      symbols = jap_romaji_symbols
    elif lang == 'jap_kana':
      available_cleaners = ['japanese_kana_cleaners']
      symbols = jap_kana_symbols
    else:
      raise RuntimeError('Wrong type of lang')
    if not use_eos:
      symbols = symbols[:-1]
    for text_cleaner in text_cleaners:
      if text_cleaner not in available_cleaners:
        raise RuntimeError(
          '{} is not available in {}.'.format(text_cleaner, lang)
        )

    self._lang = lang
    self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
    self._id_to_symbol = {i: s for i, s in enumerate(symbols)}
    self._symbols = symbols

  def get_symbols(self):
    return self._symbols

  def text_to_sequence(self, text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    if self._use_eos:
      sequence.append(self._symbol_to_id['<s>'])
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
      m = self._curly_re.match(text) if self._lang in ('eng', 'eng2', 'cmu') else None
      if not m:
        sequence += self._symbols_to_sequence(self._clean_text(text, cleaner_names))
        break
      sequence += self._symbols_to_sequence(self._clean_text(m.group(1), cleaner_names))
      sequence += self._arpabet_to_sequence(m.group(2))
      text = m.group(3)
    if self._use_eos:
      sequence.append(self._symbol_to_id['</s>'])
    return sequence


  def sequence_to_text(self, sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
      if symbol_id in self._id_to_symbol:
        s = self._id_to_symbol[symbol_id]
        # Enclose ARPAbet back in curly braces:
        if len(s) > 1 and s[0] == '@':
          s = '{%s}' % s[1:]
        result += s
    return result.replace('}{', ' ')


  def _clean_text(self, text, cleaner_names):
    for name in cleaner_names:
      cleaner = getattr(cleaners, name)
      if not cleaner:
        raise Exception('Unknown cleaner: %s' % name)
      text = cleaner(text)
    return text


  def _symbols_to_sequence(self, symbols):
    if self._lang == 'jap':
      symbols = symbols.split('-')
    return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]


  def _arpabet_to_sequence(self, text):
    return self._symbols_to_sequence(['@' + s for s in text.split()])


  def _should_keep_symbol(self, s):
    return s in self._symbol_to_id
