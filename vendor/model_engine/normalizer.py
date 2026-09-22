import re


class SinhalaTextNormalizer:
    """
    Stage 1 of the VoiceLK NLP Pipeline.

    Normalises raw, mixed Sinhala-English input text so it is suitable for
    Grapheme-to-Phoneme (G2P) conversion.  Handles the full operator and
    symbol vocabulary encountered in the Sri Lankan G.C.E. O/L ICT syllabus.

    Processing order (applied inside ``normalize``):
      1. expand_urls_and_ips  — web domains and dotted IP addresses
      2. expand_percentages   — numeric percentage expressions
      3. expand_symbols       — mathematical / logical / relational operators
      4. expand_numbers       — standalone digit sequences
      5. remove_noise         — emojis and undefined special characters
      6. pad_punctuation      — decouple punctuation from valid tokens
    """

    def __init__(self):
        # Symbol → Sinhala word mapping (sorted longest-first during expansion)
        self.symbols_map = {
            # Logical operators
            "&&": " සහ ",
            "||": " හෝ ",
            "!":  " නොවේ ",

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

    def expand_urls_and_ips(self, text: str) -> str:
        """Converts web domains and dotted IP addresses to spoken Sinhala."""
        for symbol, word in self.url_map.items():
            text = re.sub(re.escape(symbol), word, text, flags=re.IGNORECASE)
        # Replace dots between digit groups (e.g. 192.168) with spoken form
        text = re.sub(r"(?<=\d)\.(?=\d)", " ඩොට් ", text)
        return text

    def expand_percentages(self, text: str) -> str:
        """Converts numeric percentage expressions to natural Sinhala phrasing."""
        def replace_pct(match):
            num_word = self.get_sinhala_number(match.group(1))
            return f" සියයට {num_word}ක් "

        text = re.sub(r"(\d+)\s*%", replace_pct, text)
        text = text.replace("%", " ප්‍රතිශතය ")
        return text

    def expand_symbols(self, text: str) -> str:
        """Converts mathematical and logical symbols to their Sinhala equivalents."""
        # Sort longest-first to prevent partial matches (== before =, != before !)
        sorted_symbols = sorted(
            self.symbols_map.items(), key=lambda x: len(x[0]), reverse=True
        )
        for symbol, word in sorted_symbols:
            text = text.replace(symbol, word)
        return text

    def expand_numbers(self, text: str) -> str:
        """Replaces standalone digit sequences with Sinhala cardinal words."""
        return re.sub(r"\b\d+\b", lambda m: self.get_sinhala_number(m.group()), text)

    def remove_noise(self, text: str) -> str:
        """
        Strips emojis and undefined special characters.

        Retains: Sinhala Unicode block (U+0D80–U+0DFF), Zero-Width Joiner
        (U+200D) for conjunct letters, ASCII letters, whitespace, and common
        punctuation marks.
        """
        text = re.sub(r"[^\u0D80-\u0DFFa-zA-Z\s.,?!\'\"\u200D]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def pad_punctuation(self, text: str) -> str:
        """Separates punctuation marks from words to ensure clean tokenization."""
        return re.sub(r"([.,?!])", r" \1 ", text)

    def normalize(self, text: str) -> str:
        """Executes the full preprocessing pipeline in the correct logical order."""
        text = self.expand_urls_and_ips(text)
        text = self.expand_percentages(text)
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

    def _sinhala_millions(self, millions: int, remainder: int) -> str:
        """Forms Sinhala million-scale words (e.g. එක් මිලියන හතළිස් අට දහස්)."""
        attr = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        if millions == 1:
            prefix = "එක් මිලියන" if remainder else "මිලියන"
        elif millions < 10:
            prefix = f"{attr[millions]} මිලියන"
        else:
            prefix = f"{self.get_sinhala_number(str(millions))} මිලියන"
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
            return self._sinhala_millions(millions, rem)
        return " ".join(self.get_sinhala_number(d) for d in num_str if d != "0") or "බිංදුව"