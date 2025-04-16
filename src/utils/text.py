import re
from typing import List, Dict

from g2p_en import G2p

# ARPABET inventory (no stress markers)
ARPABET_PHONEMES = [
    "AA","AE","AH","AO","AW","AY",
    "B","CH","D","DH","EH","ER","EY",
    "F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P",
    "R","S","SH","T","TH","UH","UW",
    "V","W","Y","Z","ZH"
]

# Padding and end-of-sequence tokens chosen to not overlap phoneme symbols
PAD_TOKEN = "_"  # blank padding
EOS_TOKEN = "~"  # end of sequence

# Mapping from symbols to IDs
_SYMBOLS: List[str] = [PAD_TOKEN, EOS_TOKEN] + ARPABET_PHONEMES
_symbol_to_id: Dict[str,int] = {s:i for i,s in enumerate(_SYMBOLS)}

class TextProcessor:
    """
    Pure G2P-based converter from raw text to phoneme ID sequences.
    """
    def __init__(self):
        # Single G2P instance (language-agnostic)
        self.g2p = G2p()

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        # keep letters, numbers, apostrophes, spaces
        return re.sub(r"[^a-z0-9' ]+", "", text).strip()

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Normalize and apply G2P to every token in text.
        """
        normalized = self._normalize(text)
        phonemes: List[str] = []
        for word in normalized.split():
            # Get phoneme predictions from G2P
            preds = self.g2p(word)
            
            # Process phonemes - g2p_en outputs include stress markers like 'AE1'
            # Strip digits and keep only valid ARPABET phonemes
            for p in preds:
                # Remove digits (stress markers)
                clean_p = ''.join([c for c in p if not c.isdigit()])
                
                # Only add valid ARPABET phonemes
                if clean_p in ARPABET_PHONEMES:
                    phonemes.append(clean_p)
                    
        return phonemes

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """
        Append EOS and map phonemes to integer IDs.
        """
        seq = []
        for p in phonemes + [EOS_TOKEN]:
            idx = _symbol_to_id.get(p)
            if idx is None:
                raise KeyError(f"Unknown phoneme '{p}'")
            seq.append(idx)
        return seq

    def text_to_sequence(self, text: str) -> List[int]:
        """
        One-step conversion: text -> phoneme IDs.
        """
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_ids(phonemes)