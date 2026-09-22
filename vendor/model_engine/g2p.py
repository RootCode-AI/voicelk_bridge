import os
import json
import re

import eng_to_ipa as ipa
from sinling import SinhalaTokenizer


class CodeSwitchedG2P:
    """
    Stage 2 of the VoiceLK NLP Pipeline — Code-Switched Grapheme-to-Phoneme (G2P).

    Detects language boundaries in mixed Sinhala-English text (produced by
    Stage 1 normalisation) and routes each segment to the appropriate phoneme
    conversion module:

      - English tokens  → ``eng_to_ipa`` library, with lexicon overrides and
                          smart acronym spelling (e.g. RAM → ɑːr eɪ em).
      - Sinhala tokens  → rule-based consonant/vowel decomposition, with
                          ``lexicon.json`` override for ICT domain terms.

    The two streams are merged into a single, unified IPA string that is
    passed directly to the VITS acoustic model.

    IPA output is plain, character-level (phonemes are concatenated with no
    separator within a word, words separated by a single space) — this keeps
    training simple with Coqui's default character-level tokenizer, no custom
    phoneme-token tokenizer required.
    """

    def __init__(self):
        self.tokenizer = SinhalaTokenizer()
        self.en_g2p = ipa

        lexicon_path = os.path.join(os.path.dirname(__file__), "lexicon.json")
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                lexicon_data = json.load(f)
                self.en_lexicon = lexicon_data.get("en_lexicon", {})
                self.si_lexicon = lexicon_data.get("si_lexicon", {})
        except Exception:
            self.en_lexicon = {}
            self.si_lexicon = {}

        # Acronyms marked read_as='word' in the lexicon are pronounced as words
        self.word_acronyms = {
            key
            for key, entry in self.en_lexicon.items()
            if entry.get("read_as") == "word" and key.isalpha() and len(key) <= 5
        }

        # IPA for each English letter name (used when spelling acronyms)
        self.letter_ipa = {
            "a": "eɪ",  "b": "biː", "c": "siː",  "d": "diː", "e": "iː",
            "f": "ef",  "g": "dʒiː","h": "eɪtʃ", "i": "aɪ",  "j": "dʒeɪ",
            "k": "keɪ", "l": "el",  "m": "em",    "n": "en",  "o": "oʊ",
            "p": "piː", "q": "kjuː","r": "ɑːr",   "s": "es",  "t": "tiː",
            "u": "juː", "v": "viː", "w": "dʌbəl.juː", "x": "eks",
            "y": "waɪ", "z": "ziː",
        }

        # Sinhala consonant → IPA base phoneme
        self.si_consonants = {
            "ක": "k",  "ඛ": "kh", "ග": "g",  "ඝ": "gh", "ඞ": "ng",
            "ච": "tʃ", "ඡ": "tʃh","ජ": "dʒ", "ඣ": "dʒh","ඤ": "ɲ",
            "ට": "ʈ",  "ඨ": "ʈh", "ඩ": "ɖ",  "ඪ": "ɖh", "ණ": "ɳ",
            "ත": "t̪",  "ථ": "t̪h", "ද": "d̪",  "ධ": "d̪h", "න": "n",
            "ප": "p",  "ඵ": "ph", "බ": "b",  "භ": "bh", "ම": "m",
            "ය": "j",  "ර": "r",  "ල": "l",  "ව": "ʋ",
            "ශ": "ʃ",  "ෂ": "ʂ",  "ස": "s",  "හ": "h",  "ළ": "ɭ",  "ෆ": "f",
            # Prenasalized ("half-nasal") consonants — common in native
            # vocabulary (e.g. "කුඹුර" = k-u-mb-u-r-a) and previously missing,
            # which silently dropped the sound entirely.
            "ඟ": "ŋg", "ඦ": "ɲdʒ", "ඬ": "ɳɖ", "ඳ": "nd̪", "ඹ": "mb",
        }

        # Sinhala independent vowel → IPA
        self.si_vowels = {
            "අ": "a",  "ආ": "aː", "ඇ": "æ",  "ඈ": "æː",
            "ඉ": "i",  "ඊ": "iː", "උ": "u",  "ඌ": "uː",
            "එ": "e",  "ඒ": "eː", "ඔ": "o",  "ඕ": "oː",
        }

        # Sinhala vowel diacritics (attached to a consonant base)
        self.si_modifiers = {
            "්": "",   "ා": "aː", "ැ": "æ",  "ෑ": "æː",
            "ි": "i",  "ී": "iː", "ු": "u",  "ූ": "uː",
            "ෙ": "e",  "ේ": "eː", "ො": "o",  "ෝ": "oː",
            "ෞ": "au", "ෛ": "ai",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spell_as_letters(self, text: str) -> str:
        """Spells an uppercase acronym letter-by-letter (e.g. MB → em biː)."""
        return " ".join(
            self.letter_ipa[ch] for ch in text.lower() if ch in self.letter_ipa
        )

    def _process_english(self, text: str) -> str:
        """
        Converts an English token to IPA.

        Priority order:
          1. Domain lexicon override (lexicon.json → en_lexicon)
          2. Single letter → letter-name IPA
          3. ALL-CAPS word acronym → pronounce as word via eng_to_ipa
          4. ALL-CAPS other acronym → spell letter-by-letter
          5. General case → eng_to_ipa library
        """
        word_lower = text.lower()

        if word_lower in self.en_lexicon:
            return self.en_lexicon[word_lower].get("ipa", "")

        if len(text) == 1 and word_lower in self.letter_ipa:
            return self.letter_ipa[word_lower]

        if text.isupper() and text.isalpha() and len(text) >= 2:
            if word_lower in self.word_acronyms:
                return self.en_g2p.convert(text).replace("*", "")
            return self._spell_as_letters(text)

        return self.en_g2p.convert(text).replace("*", "")

    def _process_sinhala_word(self, word: str) -> str:
        """
        Converts a Sinhala token to IPA.

        Uses the ``si_lexicon`` for ICT domain overrides; falls back to
        rule-based consonant-modifier decomposition.
        """
        if word in self.si_lexicon:
            return self.si_lexicon[word].get("ipa", "")

        phonemes = []
        i, length = 0, len(word)

        while i < length:
            char = word[i]
            if char in self.si_consonants:
                base = self.si_consonants[char]
                if i + 1 < length and word[i + 1] in self.si_modifiers:
                    phonemes.append(base + self.si_modifiers[word[i + 1]])
                    i += 1
                else:
                    phonemes.append(base + "a")
            elif char in self.si_vowels:
                phonemes.append(self.si_vowels[char])
            i += 1

        return "".join(phonemes)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_unified_ipa(self, text: str) -> str:
        """
        Tokenises *text* with the Sinling tokeniser and routes each token to
        the appropriate G2P module.

        Returns a single space-joined IPA string covering all tokens,
        preserving punctuation in its original position.
        """
        tokens = self.tokenizer.tokenize(text)
        unified_ipa = []

        for token in tokens:
            if re.match(r"^[a-zA-Z]+$", token):
                unified_ipa.append(self._process_english(token))
            elif re.match(r"^[඀-෿‍]+$", token):
                unified_ipa.append(self._process_sinhala_word(token))
            else:
                # Retain punctuation and whitespace tokens unchanged
                unified_ipa.append(token)

        return " ".join(unified_ipa).strip()
