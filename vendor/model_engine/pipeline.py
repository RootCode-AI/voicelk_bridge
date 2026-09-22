from normalizer import SinhalaTextNormalizer
from g2p import CodeSwitchedG2P


class TextProcessingPipeline:
    """
    End-to-end NLP pipeline for the VoiceLK ML Service.

    Chains two sequential stages:
      Stage 1 — SinhalaTextNormalizer : Cleans and expands raw input text.
      Stage 2 — CodeSwitchedG2P       : Converts normalised text to a
                                        unified IPA phoneme sequence.
    """

    def __init__(self):
        self.normalizer = SinhalaTextNormalizer()
        self.g2p_engine = CodeSwitchedG2P()

    def process(self, raw_text: str) -> dict:
        """
        Runs the full text-processing pipeline on *raw_text*.

        Returns:
            dict with keys:
              - raw_input       : original, unmodified input string
              - normalized_text : Stage 1 output (cleaned / expanded text)
              - ipa_sequence    : Stage 2 output (unified IPA phoneme string)
        """
        normalized_text = self.normalizer.normalize(raw_text)
        ipa_sequence = self.g2p_engine.generate_unified_ipa(normalized_text)

        return {
            "raw_input": raw_text,
            "normalized_text": normalized_text,
            "ipa_sequence": ipa_sequence,
        }