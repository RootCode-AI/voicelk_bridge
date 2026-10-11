import os
import json
import re

import eng_to_ipa as ipa
from sinling import SinhalaTokenizer

# Every IPA piece eng_to_ipa can emit: the CMU phones it maps to IPA (see its
# cmu_to_ipa), the CMU consonants it passes through unchanged, and its stress
# marks.  Affricates are listed as this module writes them (tʃ / dʒ).
ENG_TO_IPA_SYMBOLS = (
    "ə eɪ ɑ æ ɔ aʊ aɪ tʃ ð ɛ ər h ɪ dʒ ŋ oʊ ɔɪ ʃ θ ʊ u ʒ i j "
    "b d f g k l m n p r s t v w z ˈ ˌ"
).split()

# Rough spelling → sound rules for English words neither the lexicon nor
# eng_to_ipa knows (multi-letter spellings first); letters not listed keep
# their own symbol (b d f g h k l m n p r s t v w z).
LETTER_TO_SOUND = {
    "tch": "tʃ", "sh": "ʃ", "ch": "tʃ", "th": "θ", "ph": "f", "ck": "k",
    "ng": "ŋ", "qu": "kw", "ee": "i", "ea": "i", "oo": "u", "ai": "eɪ",
    "ay": "eɪ", "ou": "aʊ", "ow": "oʊ", "oa": "oʊ",
    "a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɑ", "u": "ʌ", "y": "i",
    "c": "k", "x": "ks", "q": "k", "j": "dʒ",
}
LETTER_TO_SOUND_PATTERN = re.compile(
    "|".join(sorted(LETTER_TO_SOUND, key=len, reverse=True))
)


class CodeSwitchedG2P:
    """
    Stage 2 of the VoiceLK NLP Pipeline — Code-Switched Grapheme-to-Phoneme (G2P).

    Detects language boundaries in mixed Sinhala-English text (produced by
    Stage 1 normalisation) and routes each segment to the appropriate phoneme
    conversion module:

      - English tokens  → ``eng_to_ipa`` library, with lexicon overrides and
                          smart acronym spelling (e.g. RAM → ɑr eɪ ɛm).
      - Sinhala tokens  → rule-based consonant/vowel decomposition with
                          schwa rules for everyday Sinhala, with
                          ``lexicon.json`` override for ICT domain terms.

    Both languages share one symbol set: eng_to_ipa's conventions for English
    vowels, and tʃ / dʒ for the affricates.

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

        # IPA for each English letter name (used when spelling acronyms),
        # written in the same symbol set eng_to_ipa uses for English words
        self.letter_ipa = {
            "a": "eɪ",  "b": "bi",  "c": "si",   "d": "di",  "e": "i",
            "f": "ɛf",  "g": "dʒi", "h": "eɪtʃ", "i": "aɪ",  "j": "dʒeɪ",
            "k": "keɪ", "l": "ɛl",  "m": "ɛm",   "n": "ɛn",  "o": "oʊ",
            "p": "pi",  "q": "kju", "r": "ɑr",   "s": "ɛs",  "t": "ti",
            "u": "ju",  "v": "vi",  "w": "dʌbəlju", "x": "ɛks",
            "y": "waɪ", "z": "zi",
        }

        # Sinhala consonant → IPA base phoneme.  The aspirated letters
        # (mahaprana: ඛ ඝ ඡ ඣ ඨ ඪ ථ ධ ඵ භ) are spoken unaspirated in everyday
        # Sinhala, so they share the plain consonant's phoneme, and ෂ is
        # spoken as ශ.
        self.si_consonants = {
            "ක": "k",  "ඛ": "k",  "ග": "g",  "ඝ": "g",  "ඞ": "ŋ",
            "ච": "tʃ", "ඡ": "tʃ", "ජ": "dʒ", "ඣ": "dʒ", "ඤ": "ɲ",
            "ට": "ʈ",  "ඨ": "ʈ",  "ඩ": "ɖ",  "ඪ": "ɖ",  "ණ": "ɳ",
            "ත": "t̪",  "ථ": "t̪",  "ද": "d̪",  "ධ": "d̪",  "න": "n",
            "ප": "p",  "ඵ": "p",  "බ": "b",  "භ": "b",  "ම": "m",
            "ය": "j",  "ර": "r",  "ල": "l",  "ව": "ʋ",
            "ශ": "ʃ",  "ෂ": "ʃ",  "ස": "s",  "හ": "h",  "ළ": "ɭ",  "ෆ": "f",
            "ඥ": "gn",
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
            "ඍ": "ru", "ඎ": "ruː",
        }

        # Signs that follow a complete syllable (the consonant keeps its
        # inherent vowel): anusvara as in "සිංහල" = s-i-ŋ-h-a-l-a, visarga
        self.si_syllable_finals = {"ං": "ŋ", "ඃ": "h"}

        # Sinhala vowel diacritics (attached to a consonant base)
        self.si_modifiers = {
            "්": "",   "ා": "aː", "ැ": "æ",  "ෑ": "æː",
            "ි": "i",  "ී": "iː", "ු": "u",  "ූ": "uː",
            "ෙ": "e",  "ේ": "eː", "ො": "o",  "ෝ": "oː",
            "ෞ": "au", "ෛ": "ai", "ෘ": "ru", "ෲ": "ruː",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spell_as_letters(self, text: str) -> str:
        """Spells an uppercase acronym letter-by-letter (e.g. MB → ɛm bi)."""
        return " ".join(
            self.letter_ipa[ch] for ch in text.lower() if ch in self.letter_ipa
        )

    def _eng_to_ipa(self, text: str) -> str:
        """
        eng_to_ipa output with its affricate ligatures written as the
        two-symbol tʃ / dʒ that the Sinhala side and the lexicon use, so the
        model learns one symbol sequence per sound.
        """
        return self.en_g2p.convert(text).replace("ʧ", "tʃ").replace("ʤ", "dʒ")

    def _process_english(self, text: str) -> str:
        """
        Converts an English token to IPA.

        Priority order:
          1. Domain lexicon override (lexicon.json → en_lexicon)
          2. Single letter → letter-name IPA
          3. ALL-CAPS word acronym → pronounce as word via eng_to_ipa
          4. ALL-CAPS other acronym → spell letter-by-letter
          5. General case → eng_to_ipa library; a camelCase word it does not
             know is split and each part converted on its own
          6. Still unknown → _guess_english, so the spelling never reaches the
             model as raw letters
        """
        word_lower = text.lower()

        if word_lower in self.en_lexicon:
            return self.en_lexicon[word_lower].get("ipa", "")

        if len(text) == 1 and word_lower in self.letter_ipa:
            return self.letter_ipa[word_lower]

        if text.isupper() and text.isalpha() and len(text) >= 2:
            if word_lower in self.word_acronyms:
                return self._eng_to_ipa(text).replace("*", "")
            return self._spell_as_letters(text)

        converted = self._eng_to_ipa(text)
        if "*" not in converted:
            return converted

        if "'" in text:
            # An apostrophe word eng_to_ipa does not know: a possessive is the
            # word plus its -s ending, anything else is read without the mark
            base, _, suffix = text.partition("'")
            if suffix.lower() == "s":
                return self._process_english(base) + self._possessive_ending(self._process_english(base))
            return self._process_english(text.replace("'", ""))

        # eng_to_ipa marks unknown words with "*" and echoes the spelling;
        # a camelCase identifier (setState, openWeatherAPI) is read part by part
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+", text)
        if len(parts) > 1:
            return " ".join(self._process_english(part) for part in parts)
        return self._guess_english(word_lower)

    @staticmethod
    def _possessive_ending(base_ipa: str) -> str:
        """English 's: /ɪz/ after a sibilant, /s/ after another voiceless sound, else /z/."""
        if re.search(r"(s|z|ʃ|ʒ|tʃ|dʒ)$", base_ipa):
            return "ɪz"
        if re.search(r"(p|t|k|f|θ)$", base_ipa):
            return "s"
        return "z"

    def _known_english(self, word: str):
        """IPA for *word* from the lexicon or eng_to_ipa, or None if neither knows it."""
        if word in self.en_lexicon:
            return self.en_lexicon[word].get("ipa", "")
        converted = self._eng_to_ipa(word)
        return None if "*" in converted else converted

    def _guess_english(self, word: str) -> str:
        """
        Last resort for an English word nobody knows.  A compound of two known
        words (smartwatches = smart + watches) is read as those words;
        anything else is read with plain letter-to-sound rules, which is rough
        but keeps every symbol inside the model's vocabulary.
        """
        for cut in range(len(word) - 3, 2, -1):
            left, right = self._known_english(word[:cut]), self._known_english(word[cut:])
            if left and right:
                return f"{left} {right}"

        onset = ""
        if word.startswith("y"):
            onset, word = "j", word[1:]          # yes, yaml: a consonant, not /i/
        return onset + LETTER_TO_SOUND_PATTERN.sub(lambda m: LETTER_TO_SOUND[m.group()], word)

    def _process_sinhala_word(self, word: str) -> str:
        """
        Converts a Sinhala token to IPA.

        Uses the ``si_lexicon`` for ICT domain overrides; falls back to
        rule-based consonant-modifier decomposition followed by schwa rules.
        """
        if word in self.si_lexicon:
            return self.si_lexicon[word].get("ipa", "")

        # Each segment is [consonant, vowel, kind]; kind is "inherent" for a
        # consonant with no vowel sign, "hal" for a consonant with ්, "coda"
        # for ං / ඃ and "vowel" for anything with an explicit vowel.
        segments = []
        i, length = 0, len(word)

        while i < length:
            char = word[i]
            if char in self.si_consonants:
                base = self.si_consonants[char]
                if i + 1 < length and word[i + 1] in self.si_modifiers:
                    sign = word[i + 1]
                    kind = "hal" if sign == "්" else "vowel"
                    segments.append([base, self.si_modifiers[sign], kind])
                    i += 1
                else:
                    segments.append([base, "a", "inherent"])
            elif char in self.si_vowels:
                segments.append(["", self.si_vowels[char], "vowel"])
            elif char in self.si_syllable_finals:
                segments.append([self.si_syllable_finals[char], "", "coda"])
            elif char in self.si_modifiers and char != "්":
                # A vowel sign with no consonant before it, as in a suffix typed
                # straight after an English word ("Games" + "ුත්")
                segments.append(["", self.si_modifiers[char], "vowel"])
            i += 1

        self._apply_schwa_rules(segments)
        return "".join(consonant + vowel for consonant, vowel, _ in segments)

    @staticmethod
    def _apply_schwa_rules(segments: list) -> None:
        """
        Reduces the inherent vowel /a/ to /ə/ the way everyday Sinhala does
        (after Wasala, Weerasinghe & Gamage 2006, "Sinhala Grapheme-to-Phoneme
        Conversion and Rules for Schwa Epenthesis").  The inherent vowel stays
        /a/ when

          1. it is the first syllable of the word        (ක-රනවා → ka...)
          2. its syllable is closed by a hal consonant or
             by ං / ඃ                                    (ක-රන්-න → karannə)
          3. the next consonant is /h/                   (ස-ඳ-හා → sand̪ahaː)
          4. it follows a consonant + ර cluster          (සම්-ප්‍ර-දා-ය)

        and becomes /ə/ everywhere else, including the end of the word
        (මම → mamə, කියලා → kijəlaː).
        """
        seen_vowel = False
        for index, (consonant, _, kind) in enumerate(segments):
            if kind != "inherent":
                seen_vowel = seen_vowel or kind == "vowel"
                continue

            following = segments[index + 1] if index + 1 < len(segments) else None
            previous = segments[index - 1] if index > 0 else None
            keep = (
                not seen_vowel
                or (following is not None and following[2] in ("hal", "coda"))
                or (following is not None and following[0] == "h")
                or (consonant == "r" and previous is not None and previous[2] == "hal")
            )
            if not keep:
                segments[index][1] = "ə"
            seen_vowel = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def phoneme_inventory(self) -> set:
        """
        Every character this front-end can put in an IPA string, punctuation
        and spaces excluded.  Training builds the model vocabulary from it so
        no symbol the front-end produces is missing from the embedding table.
        """
        sources = list(self.si_consonants.values()) + list(self.si_vowels.values())
        sources += list(self.si_modifiers.values()) + list(self.si_syllable_finals.values())
        sources += list(self.letter_ipa.values()) + ["a", "ə"] + ENG_TO_IPA_SYMBOLS
        sources += [entry.get("ipa", "") for entry in self.en_lexicon.values()]
        sources += [entry.get("ipa", "") for entry in self.si_lexicon.values()]
        return {
            ch for text in sources for ch in text
            if not ch.isspace() and ch not in ".,?!'\""
        }

    @staticmethod
    def _join_contractions(tokens: list) -> list:
        """
        Sinling splits English words at the apostrophe ("don't" → don ' t);
        put a word ' word run back together so it is read as one word.
        """
        joined = []
        i = 0
        while i < len(tokens):
            if (
                i + 2 < len(tokens)
                and tokens[i + 1] == "'"
                and re.match(r"^[a-zA-Z]+$", tokens[i])
                and re.match(r"^[a-zA-Z]+$", tokens[i + 2])
            ):
                joined.append(f"{tokens[i]}'{tokens[i + 2]}")
                i += 3
            else:
                joined.append(tokens[i])
                i += 1
        return joined

    def generate_unified_ipa(self, text: str) -> str:
        """
        Tokenises *text* with the Sinling tokeniser and routes each token to
        the appropriate G2P module.

        Returns a single space-joined IPA string covering all tokens,
        preserving punctuation in its original position.
        """
        tokens = self._join_contractions(self.tokenizer.tokenize(text))
        unified_ipa = []

        for token in tokens:
            if re.match(r"^[a-zA-Z]+'[a-zA-Z]+$", token):
                unified_ipa.append(self._process_english(token))
                continue
            # A token can mix scripts when a Sinhala suffix is typed straight
            # after an English word ("Microsoftලාගේ"); each run is read in its
            # own language
            for piece in re.findall(r"[a-zA-Z]+|[඀-෿‍]+|[^a-zA-Z඀-෿‍]+", token) or [token]:
                if re.match(r"^[a-zA-Z]+$", piece):
                    unified_ipa.append(self._process_english(piece))
                elif re.match(r"^[඀-෿‍]+$", piece):
                    unified_ipa.append(self._process_sinhala_word(piece))
                else:
                    # Retain punctuation and whitespace tokens unchanged
                    unified_ipa.append(piece)

        return " ".join(unified_ipa).strip()
