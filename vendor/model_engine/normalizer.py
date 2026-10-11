import re
import unicodedata


class SinhalaTextNormalizer:
    """
    Stage 1 of the VoiceLK NLP Pipeline.

    Normalises raw, mixed Sinhala-English input text so it is suitable for
    Grapheme-to-Phoneme (G2P) conversion.  Handles the full operator and
    symbol vocabulary encountered in the Sri Lankan G.C.E. O/L ICT syllabus.

    Processing order (applied inside ``normalize``):
      1. remove_thousands_separators — 1,000 → 1000
      2. expand_urls_and_ips  — web domains and dotted IP addresses / versions
      3. expand_percentages   — numeric percentage expressions, decimals included
      4. split_hyphenated_words — hyphen inside a word is a space, not minus
      5. expand_slashes       — dates and word pairs; other "/" is division
      6. expand_symbols       — mathematical / logical / relational operators
      7. expand_numbers       — integers and decimals, including those against letters
      8. remove_noise         — emojis and undefined special characters
      9. pad_punctuation      — decouple punctuation from valid tokens
    """

    def __init__(self):
        # Symbol → Sinhala word mapping (sorted longest-first during expansion)
        self.symbols_map = {
            # Logical operators
            "&&": " සහ ",
            "||": " හෝ ",
            # A lone "!" is only NOT when written as an operator — see
            # expand_symbols; after a word it is an exclamation mark

            # Relational operators
            "<=": " කුඩායි හෝ සමානයි ",
            ">=": " විශාලයි හෝ සමානයි ",
            "<>": " අසමානයි ",
            "!=": " අසමානයි ",
            "==": " සමානයි ",
            "<":  " කුඩායි ",
            ">":  " විශාලයි ",

            # Arithmetic operators
            "+": " ධන ",
            "-": " සෘණ ",
            "*": " ගුණ කිරීම ",
            "/": " බෙදීම ",
            "^": " බලය ",

            # Assignment and miscellaneous
            "=":  " සමානයි ",
            "&":  " සහ ",
            "@":  " ඇට් ",
            "$":  " ඩොලර් ",
            "#":  " හැෂ් ",
        }

        self.url_map = {
            "www.":  "ඩබ්ලිව් ඩබ්ලිව් ඩබ්ලිව් ඩොට් ",
            ".com":  " ඩොට් කොම් ",
            ".lk":   " ඩොට් එල් කේ ",
            ".org":  " ඩොට් ඕර්ජී ",
            ".edu":  " ඩොට් එඩියු ",
            ".gov":  " ඩොට් ගව් ",
        }

    # ------------------------------------------------------------------
    # Public expansion methods
    # ------------------------------------------------------------------

    def remove_thousands_separators(self, text: str) -> str:
        """Joins digit groups written with thousands separators (1,000 → 1000)."""
        return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)

    def expand_urls_and_ips(self, text: str) -> str:
        """
        Converts web domains and dotted digit groups to spoken Sinhala.  Two or
        more dots (192.168.1.1, version 1.2.3) are read as "ඩොට්"; a single dot
        between digits is a decimal point and is left to expand_numbers.
        """
        # The protocol is not read aloud: "https://www.x.lk" is read from "www"
        text = re.sub(r"\b[A-Za-z]+://", " ", text)
        for symbol, word in self.url_map.items():
            text = re.sub(re.escape(symbol), word, text, flags=re.IGNORECASE)
        text = re.sub(r"\d+(?:\.\d+){2,}", lambda m: m.group().replace(".", " ඩොට් "), text)
        return text

    def expand_percentages(self, text: str) -> str:
        """Converts numeric percentage expressions to natural Sinhala phrasing."""
        def replace_pct(match):
            num_word = self.number_to_words(match.group(1))
            return f" සියයට {num_word}ක් "

        text = re.sub(r"(\d+(?:\.\d+)?)\s*%", replace_pct, text)
        text = text.replace("%", " ප්‍රතිශතය ")
        return text

    def split_hyphenated_words(self, text: str) -> str:
        """
        Turns a hyphen inside a word into a space, so compounds such as
        "compile-time", "ඊ-කසළ" or "COVID-19" are read as words rather than
        with "සෘණ".  A hyphen after a single letter ("i-1") stays minus.
        """
        text = re.sub(r"(?<=[A-Za-z඀-෿])-(?=[A-Za-z඀-෿])", " ", text)
        return re.sub(r"(?<=[A-Za-z]{2})-(?=\d)", " ", text)

    def expand_slashes(self, text: str) -> str:
        """
        Reads "/" by context: a date (2024/10/11) and a pair of words (km/h,
        TCP/IP) are read without it; anything else is division ("බෙදීම").
        """
        text = re.sub(r"\b(\d{1,4})/(\d{1,2})/(\d{1,4})\b", r"\1 \2 \3", text)
        # A word on the left (two letters or more) makes it a pair; "x/y" is division
        return re.sub(r"(?<=[A-Za-z඀-෿]{2})/(?=[A-Za-z඀-෿])", " ", text)

    def expand_symbols(self, text: str) -> str:
        """Converts mathematical and logical symbols to their Sinhala equivalents."""
        # Sort longest-first to prevent partial matches (== before =, != before !)
        sorted_symbols = sorted(
            self.symbols_map.items(), key=lambda x: len(x[0]), reverse=True
        )
        for symbol, word in sorted_symbols:
            text = text.replace(symbol, word)
        # Logical NOT: "!" written straight before an operand ("!x", "(!done)"),
        # at the start or after a space, bracket or operator.  Any other "!"
        # ("හරි!", "!!!") stays punctuation.
        text = re.sub(r"(?:^|(?<=[\s(=&|]))!(?=[A-Za-z0-9඀-෿(])", " නොවේ ", text)
        return text

    def expand_numbers(self, text: str) -> str:
        """
        Replaces digit sequences with Sinhala cardinal words.

        Digits written against letters are read too instead of being dropped
        later as noise: against Latin letters they become a separate word
        ("4G" → "හතර G", "MP3" → "MP තුන"); a Sinhala suffix stays attached
        ("5ක්" → "පහක්", "10ට" → "දහයට").
        """
        text = re.sub(r"(?<=[A-Za-z඀-෿])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
        return re.sub(r"\d+(?:\.\d+)?", lambda m: self.number_to_words(m.group()), text)

    def number_to_words(self, num_str: str) -> str:
        """
        Sinhala words for an integer or a decimal; the digits after the point
        are read one by one ("3.14" → "තුන දශම එක හතර").
        """
        whole, _, fraction = num_str.partition(".")
        words = self.get_sinhala_number(whole)
        if fraction:
            words += " දශම " + " ".join(self.get_sinhala_number(d) for d in fraction)
        return words

    def remove_noise(self, text: str) -> str:
        """
        Strips emojis and undefined special characters.

        Retains: Sinhala Unicode block (U+0D80–U+0DFF), Zero-Width Joiner
        (U+200D) for conjunct letters, ASCII letters, whitespace, and common
        punctuation marks.  Accented Latin letters are folded to plain ASCII
        first (Düsseldorf → Dusseldorf) instead of splitting the word, and a
        typographic apostrophe (’) becomes a plain one so "don’t" stays a word.
        """
        text = text.replace("’", "'")
        text = re.sub(
            r"[À-ɏ]",
            lambda m: unicodedata.normalize("NFKD", m.group()).encode("ascii", "ignore").decode(),
            text,
        )
        text = re.sub(r"[^\u0D80-\u0DFFa-zA-Z\s.,?!\'\"\u200D]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def pad_punctuation(self, text: str) -> str:
        """Separates punctuation marks from words to ensure clean tokenization."""
        return re.sub(r"([.,?!])", r" \1 ", text)

    def normalize(self, text: str) -> str:
        """Executes the full preprocessing pipeline in the correct logical order."""
        text = self.remove_thousands_separators(text)
        text = self.expand_urls_and_ips(text)
        text = self.expand_percentages(text)
        text = self.split_hyphenated_words(text)
        text = self.expand_slashes(text)
        text = self.expand_symbols(text)
        text = self.expand_numbers(text)
        text = self.remove_noise(text)
        text = self.pad_punctuation(text)  # Automatically decouples punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Number-to-Sinhala helpers
    # ------------------------------------------------------------------

    def _sinhala_below_100(self, n: int) -> str:
        """Returns natural Sinhala cardinal words for 1–99."""
        ones = ["", "එක", "දෙක", "තුන", "හතර", "පහ", "හය", "හත", "අට", "නවය"]
        teens = [
            "දහය", "එකොළහ", "දොළහ", "දහතුන", "දහහතර",
            "පහළොව", "දහසය", "දහහත", "දහඅට", "දහනවය",
        ]
        tens_exact  = ["", "දහය", "විස්ස", "තිහ", "හතළිහ", "පනහ", "හැට", "හැත්තෑව", "අසූව", "අනූව"]
        tens_prefix = ["", "දහ", "විසි", "තිස්", "හතළිස්", "පනස්", "හැට", "හැත්තෑ", "අසූ", "අනූ"]

        if n == 0:
            return ""
        if 10 <= n <= 19:
            return teens[n - 10]
        tens, unit = divmod(n, 10)
        parts = []
        if tens:
            parts.append(tens_exact[tens] if unit == 0 else tens_prefix[tens])
        if unit:
            parts.append(ones[unit])
        return " ".join(parts)

    def _sinhala_below_1000(self, n: int) -> str:
        """Returns natural Sinhala cardinal words for 1–999."""
        hundreds_exact  = ["", "සියය", "දෙසියය", "තුන්සියය", "හාරසියය", "පන්සියය",
                           "හයසියය", "හත්සියය", "අටසියය", "නවසියය"]
        hundreds_prefix = ["", "එකසිය", "දෙසිය", "තුන්සිය", "හාරසිය", "පන්සිය",
                           "හයසිය", "හත්සිය", "අටසිය", "නවසිය"]

        if n < 100:
            return self._sinhala_below_100(n)
        hundreds, rem = divmod(n, 100)
        parts = [hundreds_exact[hundreds] if rem == 0 else hundreds_prefix[hundreds]]
        if rem:
            parts.append(self._sinhala_below_100(rem))
        return " ".join(p for p in parts if p)

    def _sinhala_thousands(self, thousands: int, remainder: int) -> str:
        """Forms Sinhala thousand-scale words (e.g. එක්දහස් විසි හතර)."""
        attr = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        if thousands == 1:
            prefix = "එක්දහස්" if remainder else "දහස"
        elif thousands < 10:
            prefix = f"{attr[thousands]}දහස්" if remainder else f"{attr[thousands]}දහස"
        else:
            prefix = f"{self.get_sinhala_number(str(thousands))} දහස්"
        parts = [prefix]
        if remainder:
            parts.append(self._sinhala_below_1000(remainder))
        return " ".join(parts)

    def _sinhala_large(self, count: int, remainder: int, scale: str) -> str:
        """Forms Sinhala million / billion-scale words (e.g. එක් මිලියන හතළිස් අට දහස්)."""
        attr = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        if count == 1:
            prefix = f"එක් {scale}" if remainder else scale
        elif count < 10:
            prefix = f"{attr[count]} {scale}"
        else:
            prefix = f"{self.get_sinhala_number(str(count))} {scale}"
        parts = [prefix]
        if remainder:
            parts.append(self.get_sinhala_number(str(remainder)))
        return " ".join(parts)

    def get_sinhala_number(self, num_str: str) -> str:
        """
        Converts a digit string to natural Sinhala cardinal words.

        Examples:
          "0"       → "බිංදුව"
          "1024"    → "එක්දහස් විසි හතර"
          "1000000" → "මිලියන"
          "1000000000" → "බිලියන"

        A value of a trillion or more (an account or card number rather than a
        quantity) is read digit by digit, zeros included.
        """
        num = int(num_str)
        if num == 0:
            return "බිංදුව"
        if num < 1_000:
            return self._sinhala_below_1000(num)
        if num < 1_000_000:
            thousands, rem = divmod(num, 1_000)
            return self._sinhala_thousands(thousands, rem)
        if num < 1_000_000_000:
            millions, rem = divmod(num, 1_000_000)
            return self._sinhala_large(millions, rem, "මිලියන")
        if num < 1_000_000_000_000:
            billions, rem = divmod(num, 1_000_000_000)
            return self._sinhala_large(billions, rem, "බිලියන")
        return " ".join(self.get_sinhala_number(d) for d in num_str)