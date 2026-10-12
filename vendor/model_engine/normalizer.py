import re
import unicodedata


# Word runs the language of a number is judged by, and the marks that end a sentence
_SCAN_TOKEN = re.compile(r"[A-Za-z]+|[඀-෿‍]+|[.?!]|\S")

# Bound Sinhala endings written apart from the number they belong to ("35,000 ක්",
# "22 කට", "2026 යේ", "25 වැනි"); a number carrying one is read in Sinhala
_SINHALA_NUMBER_SUFFIXES = {
    "ක්", "ක", "කට", "කින්", "කුත්", "ක්ම", "කදී", "කයි",
    "ට", "ටත්", "ටම", "යි", "යේ", "යක්", "ගේ",
    "වැනි", "වෙනි", "වන",
}

# Units read after a number.  A unit the speakers spell out letter by letter
# (kg, fps, mAh) is written in capitals so the G2P spells it like any acronym.
_UNIT_WORDS = {
    "W": "watt", "w": "watt", "Wh": "watt hour", "kW": "kilowatt",
    "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz",
    "mm": "millimeter", "cm": "centimeter", "nm": "nanometer", "km": "kilometer",
    "ms": "millisecond",
    "kg": "KG", "Kg": "KG", "KG": "KG", "kgs": "KG",
    "fps": "FPS", "FPS": "FPS", "mAh": "MAH",
}
# Single-letter units are only taken when written against the number ("25W")
_UNIT_PATTERN = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)(?:(?P<attached>%s)|\s(?P<spaced>%s))(?![A-Za-z])" % (
        "|".join(sorted(_UNIT_WORDS, key=len, reverse=True)),
        "|".join(sorted((u for u in _UNIT_WORDS if len(u) > 1), key=len, reverse=True)),
    )
)

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


class SinhalaTextNormalizer:
    """
    Stage 1 of the VoiceLK NLP Pipeline.

    Normalises raw, mixed Sinhala-English input text so it is suitable for
    Grapheme-to-Phoneme (G2P) conversion.  Handles the full operator and
    symbol vocabulary encountered in the Sri Lankan G.C.E. O/L ICT syllabus.

    Processing order (applied inside ``normalize``):
      1. remove_thousands_separators — 1,000 → 1000, 4,79,000 → 479000
      2. expand_urls_and_ips  — web domains and dotted IP addresses / versions
      3. expand_percentages   — numeric percentage expressions, decimals included
      4. split_hyphenated_words — hyphen inside a word is a space, not minus
      5. expand_slashes       — dates and word pairs; other "/" is division
      6. expand_currency      — "$20" with the number
      7. expand_symbols       — mathematical / logical / relational operators
      8. expand_numbers       — integers and decimals, including those against letters
      9. remove_noise         — emojis and undefined special characters
     10. pad_punctuation      — decouple punctuation from valid tokens

    A number is read in the language its speakers use for it (see
    ``number_language``): in English next to English words ("6 month",
    "Windows 11", "25W"), in Sinhala with a Sinhala ending ("5ක්", "100%ක්")
    or among Sinhala words ("2026 බලනකොට").
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

        # Words the URL / IP expansion writes in; a number between them is
        # judged by the words around the whole address, not by them
        self.transparent_words = {
            word for spoken in self.url_map.values() for word in spoken.split()
        }

    # ------------------------------------------------------------------
    # Public expansion methods
    # ------------------------------------------------------------------

    def remove_thousands_separators(self, text: str) -> str:
        """
        Joins digit groups written with thousands separators, both the Western
        grouping (1,000 → 1000) and the Sri Lankan lakh grouping
        (4,79,000 → 479000).
        """
        text = re.sub(r"(?<![\d,])\d{1,2}(?:,\d{2})+,\d{3}(?![\d,])",
                      lambda m: m.group().replace(",", ""), text)
        return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)

    def expand_urls_and_ips(self, text: str) -> str:
        """
        Converts web domains and dotted digit groups to speech.  Two or more
        dots (192.168.1.1, version 1.2.3) are read as "ඩොට්", or as "point" in a
        version number among English words ("Android 4.2.1"); a single dot
        between digits is a decimal point and is left to expand_numbers.
        """
        # The protocol is not read aloud: "https://www.x.lk" is read from "www"
        text = re.sub(r"\b[A-Za-z]+://", " ", text)
        for symbol, word in self.url_map.items():
            text = re.sub(re.escape(symbol), word, text, flags=re.IGNORECASE)

        def replace_dotted(match):
            groups = match.group().split(".")
            is_ip = len(groups) == 4 and all(int(g) <= 255 for g in groups)
            if is_ip or self.number_language(match.string, match.start(), match.end()) == "si":
                return match.group().replace(".", " ඩොට් ")
            return match.group().replace(".", " point ")

        return re.sub(r"\d+(?:\.\d+){2,}", replace_dotted, text)

    def expand_percentages(self, text: str) -> str:
        """
        Converts numeric percentage expressions to speech: "N percent" among
        English words, otherwise Sinhala, where a written ending takes the place
        of "ක්" ("80%" → "සියයට අසූවක්", "10%ටත්" → "සියයට දහයටත්").
        """
        def replace_pct(match):
            number, ending = match.group(1), match.group(2)
            if not ending and self.number_language(match.string, match.start(), match.end()) == "en":
                return f" {self.english_number(number)} percent "
            return f" සියයට {self.number_to_words(number)}{ending or 'ක්'} "

        text = re.sub(r"(\d+(?:\.\d+)?)\s*%([඀-෿‍]*)", replace_pct, text)
        text = text.replace("%", " ප්‍රතිශතය ")
        return text

    def split_hyphenated_words(self, text: str) -> str:
        """
        Turns a hyphen inside a word into a space, so compounds such as
        "compile-time", "ඊ-කසළ", "COVID-19" or "64-bit" are read as words
        rather than with "සෘණ".  A hyphen after a single letter ("i-1") or
        between two numbers ("5-3") stays minus.
        """
        text = re.sub(r"(?<=[A-Za-z඀-෿])-(?=[A-Za-z඀-෿])", " ", text)
        text = re.sub(r"(?<=\d)-(?=[A-Za-z]{2})", " ", text)
        return re.sub(r"(?<=[A-Za-z]{2})-(?=\d)", " ", text)

    def expand_slashes(self, text: str) -> str:
        """
        Reads "/" by context: a date (2024/10/11), "24/7" and a pair of words
        (km/h, TCP/IP) are read without it; anything else is division ("බෙදීම").
        """
        text = re.sub(r"(?<![\d/])24/7(?![\d/])", " twenty four seven ", text)
        text = re.sub(r"\b(\d{1,4})/(\d{1,2})/(\d{1,4})\b", r"\1 \2 \3", text)
        # A word on the left (two letters or more) makes it a pair; "x/y" is division
        return re.sub(r"(?<=[A-Za-z඀-෿]{2})/(?=[A-Za-z඀-෿])", " ", text)

    def expand_currency(self, text: str) -> str:
        """
        Reads a dollar amount with its number: "twenty dollars" among English
        words, otherwise "ඩොලර් විස්ස" as the symbol has always been read.
        """
        def replace_dollars(match):
            amount = match.group(1)
            if self.number_language(match.string, match.start(), match.end()) == "si":
                return f" ඩොලර් {self.number_to_words(amount)} "
            whole, _, cents = amount.partition(".")
            if len(cents) == 2 and int(cents):
                cents_words = f"{self._english_below_100(int(cents))} cent{'s' if int(cents) != 1 else ''}"
                if int(whole) == 0:
                    return f" {cents_words} "
                whole_words = self.english_number(whole)
                return f" {whole_words} dollar{'s' if int(whole) != 1 else ''} {cents_words} "
            return f" {self.english_number(amount)} dollar{'s' if amount != '1' else ''} "

        return re.sub(r"\$\s?(\d+(?:\.\d+)?)s?(?![A-Za-z\d])", replace_dollars, text)

    def expand_symbols(self, text: str) -> str:
        """Converts mathematical and logical symbols to their Sinhala equivalents."""
        # A "+" closing a number ("2000+", "P1+") is "plus", not addition
        text = re.sub(r"(?<=\d)\+(?![\w(+])", " plus ", text)
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
        Replaces digit sequences with words in the language they are spoken in
        (see ``number_language``).

        Forms with a fixed English reading are taken first: units ("25W" →
        "twenty five watt"), ordinals ("28th"), decades ("90s"), video
        resolutions ("1080p" → "one zero eight zero p"), dimensions ("16x10" → "sixteen
        by ten"), clock times and ratios ("7:30 p.m.", "16:9") and degrees.
        Digits written against letters are read too instead of being dropped
        later as noise: against Latin letters they become a separate word
        ("4G" → "four G", "MP3" → "MP three"); a Sinhala suffix stays attached
        ("5ක්" → "පහක්", "10ට" → "දහයට").
        """
        text = self._expand_units(text)
        text = self._expand_ordinals_and_decades(text)
        text = self._expand_resolutions_and_dimensions(text)
        text = self._expand_times_and_ratios(text)
        text = self._expand_degrees(text)

        def replace_number(match):
            source, start, end = match.string, match.start(), match.end()
            if self.number_language(source, start, end) == "en":
                words = self.english_number(match.group())
            else:
                words = self.number_to_words(match.group())
            # Latin letters on either side become a separate word; a Sinhala
            # ending after the number stays attached to it
            before = " " if start and re.match(r"[A-Za-z඀-෿]", source[start - 1]) else ""
            after = " " if end < len(source) and re.match(r"[A-Za-z]", source[end]) else ""
            return f"{before}{words}{after}"

        return re.sub(r"\d+(?:\.\d+)?", replace_number, text)

    def number_language(self, text: str, start: int, end: int) -> str:
        """
        "en" or "si": the language the number at text[start:end] is read in.

        Measured against how the speakers in the recorded corpus read numbers:

          1. A Sinhala ending, attached or written apart ("5ක්", "35,000 ක්",
             "2026 යේ"), is Sinhala grammar — Sinhala ("inches 100ක්" → සීයක්).
          2. Latin letters against the number ("R24", "25W", "4K") — English.
          3. An English word as the nearest word on either side, looking past
             other numbers and punctuation but not past the end of a sentence
             ("6 month", "Microsoft 365 පිටිපස්සෙත්", "ready, 3, 2, 1") — English.
          4. Anything else — Sinhala ("2026 බලනකොට", "i - 1", a bare number).
        """
        if end < len(text) and re.match(r"[඀-෿]", text[end]):
            return "si"
        if (start and re.match(r"[A-Za-z]", text[start - 1])) or \
                (end < len(text) and re.match(r"[A-Za-z]", text[end])):
            return "en"

        following = self._nearest_word(_SCAN_TOKEN.finditer(text, end), check_suffix=True)
        if following == "suffix":
            return "si"
        preceding = self._nearest_word(reversed(list(_SCAN_TOKEN.finditer(text[:start]))))
        return "en" if "en" in (preceding, following) else "si"

    def _nearest_word(self, tokens, check_suffix: bool = False):
        """'en', 'si' or 'suffix' for the first word among *tokens*, None at a sentence end."""
        for position, token in enumerate(tokens):
            word = token.group()
            if word in ".?!":
                return None
            if re.match(r"[A-Za-z]", word):
                return "en"
            if re.match(r"[඀-෿]", word):
                if check_suffix and position == 0 and word in _SINHALA_NUMBER_SUFFIXES:
                    return "suffix"
                if word not in self.transparent_words:
                    return "si"
        return None

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
        text = re.sub(r"[^඀-෿a-zA-Z\s.,?!\'\"‍]", " ", text)
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
        text = self.expand_currency(text)
        text = self.expand_symbols(text)
        text = self.expand_numbers(text)
        text = self.remove_noise(text)
        text = self.pad_punctuation(text)  # Automatically decouples punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Number forms with a fixed English reading
    # ------------------------------------------------------------------

    def _expand_units(self, text: str) -> str:
        """A number with a unit after it, read in English: "25W" → "twenty five watt"."""
        def replace_unit(match):
            unit = match.group("attached") or match.group("spaced")
            return f" {self.english_number(match.group(1))} {_UNIT_WORDS[unit]} "

        return _UNIT_PATTERN.sub(replace_unit, text)

    def _expand_ordinals_and_decades(self, text: str) -> str:
        """English ordinals ("28th" → "twenty eighth") and decades ("1950s" → "nineteen fifties")."""
        text = re.sub(
            r"(?<![\d.])(\d+)(?:st|nd|rd|th)(?![A-Za-z])",
            lambda m: f" {self.english_ordinal(m.group(1))} ", text, flags=re.IGNORECASE,
        )
        return re.sub(
            r"(?<![\d.])(\d*0)s(?![A-Za-z])",
            lambda m: f" {self.english_decade(m.group(1))} ", text,
        )

    def _expand_resolutions_and_dimensions(self, text: str) -> str:
        """
        Video resolutions digit by digit ("1080p" → "one zero eight zero p"),
        then figures read in pairs: four-digit graphics and mobile chip models
        ("RTX 4050" → "RTX forty fifty", "Dimensity 6300") and laptop CPU models
        with their suffix
        ("8640HS" → "eighty six forty HS").  "16x10" → "sixteen by ten".
        """
        # The project's ICT term list reads resolutions digit by digit
        # ("720p" → සෙවන් ටූ සීරෝ පී)
        text = re.sub(
            r"(?<![\d.])(360|480|720|1080|1440|2160|4320)p(?![A-Za-z])",
            lambda m: " " + " ".join(_EN_ONES[int(d)] for d in m.group(1)) + " p ", text,
        )
        text = re.sub(
            r"\b(RTX|GTX|RX|Dimensity)\s?(\d{4})(?![\d.]|[A-Za-z])",
            lambda m: f" {m.group(1)} {self._english_in_pairs(m.group(2))} ", text,
        )
        text = re.sub(
            r"(?<![\d.])(\d{4})(HS|HX|H|U)(?![A-Za-z])",
            lambda m: f" {self._english_in_pairs(m.group(1))} {m.group(2)} ", text,
        )
        return re.sub(
            r"(?<![\d.])(?!0[xX])(\d+(?:\.\d+)?)\s?[xX×]\s?(\d+(?:\.\d+)?)(?![\dA-Za-z]|\.\d)",
            lambda m: f" {self.english_number(m.group(1))} by {self.english_number(m.group(2))} ",
            text,
        )

    def _expand_times_and_ratios(self, text: str) -> str:
        """
        "7:30 p.m." → "seven thirty p.m."; any other "A:B" is a ratio
        ("16:9" → "sixteen is to nine"), except two-digit minutes after an hour
        ("10:30"), which is a time unless the sentence speaks of a ratio.
        """
        def replace_colon(match):
            left, right = match.group(1), match.group(2)
            following = match.string[match.end():match.end() + 12]
            around = match.string[max(0, match.start() - 30):match.end() + 30].lower()
            clock = "." not in left + right and int(left) <= 24 and len(right) == 2 and int(right) < 60
            marked = re.match(r"\s?[AaPp]\.?\s?[Mm]\b", following)
            if clock and (marked or ("ratio" not in around and "aspect" not in around)):
                minutes = int(right)
                if minutes == 0:
                    tail = "" if marked else " o'clock"
                elif minutes < 10:
                    tail = f" oh {_EN_ONES[minutes]}"
                else:
                    tail = f" {self._english_below_100(minutes)}"
                return f" {self.english_number(left)}{tail} "
            return f" {self.english_number(left)} is to {self.english_number(right)} "

        return re.sub(r"(?<![\d.:])(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)(?![\d:])", replace_colon, text)

    def _expand_degrees(self, text: str) -> str:
        """"180°" → "one hundred eighty degrees"; among Sinhala words "අංශක එකසිය අසූව"."""
        scales = {"C": ("celsius", "සෙල්සියස් "), "F": ("fahrenheit", "ෆැරන්හයිට් ")}

        def replace_degrees(match):
            number, scale = match.group(1), match.group(2)
            if self.number_language(match.string, match.start(), match.start() + len(number)) == "si" \
                    and not re.match(r"[A-Za-z]", match.string[match.end():match.end() + 1] or " "):
                return f" {scales[scale][1] if scale else ''}අංශක {self.number_to_words(number)} "
            word = "degree" if number == "1" else "degrees"
            return f" {self.english_number(number)} {word}{' ' + scales[scale][0] if scale else ''} "

        return re.sub(r"(\d+(?:\.\d+)?)\s?°\s?([CF](?![A-Za-z]))?", replace_degrees, text)

    # ------------------------------------------------------------------
    # Number-to-English helpers
    # ------------------------------------------------------------------

    def english_number(self, num_str: str) -> str:
        """
        English words for an integer or a decimal, the way the speakers read
        them: no "and" ("one hundred forty four"), "hundred" for 100 on its own
        ("100% safe"), digits after the point one by one ("2.2" → "two point
        two"), a leading zero digit by digit ("00" → "zero zero").
        """
        whole, _, fraction = num_str.partition(".")
        if whole == "100" and not fraction:
            words = "hundred"
        elif len(whole) > 1 and whole.startswith("0"):
            words = " ".join(_EN_ONES[int(d)] for d in whole)
        else:
            words = self._english_cardinal(int(whole or "0"))
        if fraction:
            words += " point " + " ".join(_EN_ONES[int(d)] for d in fraction)
        return words

    def english_ordinal(self, num_str: str) -> str:
        """"1" → "first", "28" → "twenty eighth", "10" → "tenth"."""
        words = self._english_cardinal(int(num_str)).split()
        last = words[-1]
        if last in _EN_ORDINALS:
            words[-1] = _EN_ORDINALS[last]
        elif last.endswith("y"):
            words[-1] = last[:-1] + "ieth"
        else:
            words[-1] = last + "th"
        return " ".join(words)

    def english_decade(self, num_str: str) -> str:
        """"90s" → "nineties", "1950s" → "nineteen fifties", "2000s" → "two thousands"."""
        n = int(num_str)
        if 1100 <= n <= 2099 and n % 100:
            return f"{self._english_below_100(n // 100)} {self._plural(self._english_below_100(n % 100))}"
        if 1100 <= n <= 1999:
            return f"{self._english_below_100(n // 100)} hundreds"
        return self._plural(self._english_cardinal(n))

    def _english_in_pairs(self, num_str: str) -> str:
        """Reads a four-digit figure in pairs: "4050" → "forty fifty", "6300" → "sixty three hundred"."""
        head, tail = num_str[:-2], int(num_str[-2:])
        tail_words = self._english_below_100(tail) if tail else "hundred"
        return f"{self._english_below_100(int(head))} {tail_words}"

    @staticmethod
    def _plural(words: str) -> str:
        head, _, last = words.rpartition(" ")
        last = last[:-1] + "ies" if last.endswith("y") else last + "s"
        return f"{head} {last}".strip()

    def _english_below_100(self, n: int) -> str:
        if n < 10:
            return _EN_ONES[n]
        if n < 20:
            return _EN_TEENS[n - 10]
        tens, unit = divmod(n, 10)
        return _EN_TENS[tens] + (f" {_EN_ONES[unit]}" if unit else "")

    def _english_cardinal(self, n: int) -> str:
        """Cardinal words for n; a trillion or more is read digit by digit, like the Sinhala side."""
        if n >= 1_000_000_000_000:
            return " ".join(_EN_ONES[int(d)] for d in str(n))
        if n < 100:
            return self._english_below_100(n)
        if n < 1000:
            hundreds, rest = divmod(n, 100)
            return f"{_EN_ONES[hundreds]} hundred" + (f" {self._english_below_100(rest)}" if rest else "")
        for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
            if n >= scale:
                count, rest = divmod(n, scale)
                words = f"{self._english_cardinal(count)} {name}"
                return words + (f" {self._english_cardinal(rest)}" if rest else "")
        return ""

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

    def _sinhala_attributive(self, n: int) -> str:
        """
        The form of 1–99 that is joined to the scale word after it, as in
        "දෙදහස", "විසි පන්දහස" (25,000), "දහතුන්දහස" (13,000), "තිස්දහස" (30,000).
        """
        units = ["", "එක්", "දෙ", "තුන්", "හාර", "පන්", "හය", "හත්", "අට", "නව"]
        teens = ["දස", "එකොළොස්", "දොළොස්", "දහතුන්", "දහහතර",
                 "පහළොස්", "දහසය", "දහහත්", "දහඅට", "දහනව"]
        tens_prefix = ["", "දහ", "විසි", "තිස්", "හතළිස්", "පනස්", "හැට", "හැත්තෑ", "අසූ", "අනූ"]
        if n < 10:
            return units[n]
        if n < 20:
            return teens[n - 10]
        tens, unit = divmod(n, 10)
        return tens_prefix[tens] + (f" {units[unit]}" if unit else "")

    def _sinhala_thousands(self, thousands: int, remainder: int) -> str:
        """Forms Sinhala thousand-scale words, 1,000–99,999 (e.g. එක්දහස් විසි හතර, තිස්දහස)."""
        if thousands == 1:
            prefix = "එක්දහස්" if remainder else "දහස"
        else:
            prefix = self._sinhala_attributive(thousands) + ("දහස්" if remainder else "දහස")
        parts = [prefix]
        if remainder:
            parts.append(self._sinhala_below_1000(remainder))
        return " ".join(parts)

    def _sinhala_lakhs(self, lakhs: int, remainder: int) -> str:
        """Forms Sinhala lakh-scale words, 100,000–999,999 (e.g. ලක්ෂ හතර හැත්තෑ නවදහස)."""
        if lakhs == 1 and not remainder:
            return "ලක්ෂය"
        parts = ["ලක්ෂ", self._sinhala_below_100(lakhs)]
        if remainder:
            parts.append(self.get_sinhala_number(str(remainder)))
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
          "30000"   → "තිස්දහස"
          "479000"  → "ලක්ෂ හතර හැත්තෑ නවදහස"
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
        if num < 100_000:
            thousands, rem = divmod(num, 1_000)
            return self._sinhala_thousands(thousands, rem)
        if num < 1_000_000:
            lakhs, rem = divmod(num, 100_000)
            return self._sinhala_lakhs(lakhs, rem)
        if num < 1_000_000_000:
            millions, rem = divmod(num, 1_000_000)
            return self._sinhala_large(millions, rem, "මිලියන")
        if num < 1_000_000_000_000:
            billions, rem = divmod(num, 1_000_000_000)
            return self._sinhala_large(billions, rem, "බිලියන")
        return " ".join(self.get_sinhala_number(d) for d in num_str)
