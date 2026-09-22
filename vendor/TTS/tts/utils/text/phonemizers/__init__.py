from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

# VoiceLK note: these phonemizers each require an optional third-party language package
# (bangla, gruut, jpype1/belarusian deps, korean/chinese g2p libs) that this project never
# installs, since VoiceLK's own model_engine G2P pipeline is used instead (use_phonemes=False
# everywhere in model_training/train_*.py — TTS's built-in phonemizers are never invoked at
# runtime). Each import is optional so `import TTS...` doesn't hard-fail when one of those
# extras isn't installed, matching the upstream project's own handling of JA_JP_Phonemizer.
try:
    from TTS.tts.utils.text.phonemizers.bangla_phonemizer import BN_Phonemizer
except ImportError:
    BN_Phonemizer = None

try:
    from TTS.tts.utils.text.phonemizers.belarusian_phonemizer import BEL_Phonemizer
except ImportError:
    BEL_Phonemizer = None

try:
    from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
except ImportError:
    Gruut = None

try:
    from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer
except ImportError:
    KO_KR_Phonemizer = None

try:
    from TTS.tts.utils.text.phonemizers.zh_cn_phonemizer import ZH_CN_Phonemizer
except ImportError:
    ZH_CN_Phonemizer = None

try:
    from TTS.tts.utils.text.phonemizers.ja_jp_phonemizer import JA_JP_Phonemizer
except ImportError:
    JA_JP_Phonemizer = None


PHONEMIZERS = {ESpeak.name(): ESpeak}
for _p in (Gruut, KO_KR_Phonemizer, BN_Phonemizer):
    if _p is not None:
        PHONEMIZERS[_p.name()] = _p


ESPEAK_LANGS = list(ESpeak.supported_languages().keys())
GRUUT_LANGS = list(Gruut.supported_languages()) if Gruut is not None else []


# Dict setting default phonemizers for each language
# Add Gruut languages (if available)
DEF_LANG_TO_PHONEMIZER = dict(list(zip(GRUUT_LANGS, [Gruut.name()] * len(GRUUT_LANGS)))) if Gruut is not None else {}


# Add ESpeak languages and override any existing ones
_new_dict = dict(list(zip(list(ESPEAK_LANGS), [ESpeak.name()] * len(ESPEAK_LANGS))))
DEF_LANG_TO_PHONEMIZER.update(_new_dict)


# Force default for some languages (only if that phonemizer/language is actually available)
if "en-us" in DEF_LANG_TO_PHONEMIZER:
    DEF_LANG_TO_PHONEMIZER["en"] = DEF_LANG_TO_PHONEMIZER["en-us"]
if ZH_CN_Phonemizer is not None:
    DEF_LANG_TO_PHONEMIZER["zh-cn"] = ZH_CN_Phonemizer.name()
if KO_KR_Phonemizer is not None:
    DEF_LANG_TO_PHONEMIZER["ko-kr"] = KO_KR_Phonemizer.name()
if BN_Phonemizer is not None:
    DEF_LANG_TO_PHONEMIZER["bn"] = BN_Phonemizer.name()
if BEL_Phonemizer is not None:
    DEF_LANG_TO_PHONEMIZER["be"] = BEL_Phonemizer.name()


# JA phonemizer has deal breaking dependencies like MeCab for some systems.
# So we only have it when we have it.
if JA_JP_Phonemizer is not None:
    PHONEMIZERS[JA_JP_Phonemizer.name()] = JA_JP_Phonemizer
    DEF_LANG_TO_PHONEMIZER["ja-jp"] = JA_JP_Phonemizer.name()


def get_phonemizer_by_name(name: str, **kwargs) -> BasePhonemizer:
    """Initiate a phonemizer by name

    Args:
        name (str):
            Name of the phonemizer that should match `phonemizer.name()`.

        kwargs (dict):
            Extra keyword arguments that should be passed to the phonemizer.
    """
    if name == "espeak":
        return ESpeak(**kwargs)
    if name == "gruut":
        if Gruut is None:
            raise ValueError(" ❗ You need to install the `gruut` package to use the Gruut phonemizer.")
        return Gruut(**kwargs)
    if name == "zh_cn_phonemizer":
        if ZH_CN_Phonemizer is None:
            raise ValueError(" ❗ You need to install the Chinese phonemizer dependencies (jieba, pypinyin).")
        return ZH_CN_Phonemizer(**kwargs)
    if name == "ja_jp_phonemizer":
        if JA_JP_Phonemizer is None:
            raise ValueError(" ❗ You need to install JA phonemizer dependencies. Try `pip install TTS[ja]`.")
        return JA_JP_Phonemizer(**kwargs)
    if name == "ko_kr_phonemizer":
        if KO_KR_Phonemizer is None:
            raise ValueError(" ❗ You need to install the Korean phonemizer dependencies (jamo, g2pkk).")
        return KO_KR_Phonemizer(**kwargs)
    if name == "bn_phonemizer":
        if BN_Phonemizer is None:
            raise ValueError(" ❗ You need to install the `bangla` package to use the Bangla phonemizer.")
        return BN_Phonemizer(**kwargs)
    if name == "be_phonemizer":
        if BEL_Phonemizer is None:
            raise ValueError(" ❗ You need to install the `jpype1` package to use the Belarusian phonemizer.")
        return BEL_Phonemizer(**kwargs)
    raise ValueError(f"Phonemizer {name} not found")


if __name__ == "__main__":
    print(DEF_LANG_TO_PHONEMIZER)
