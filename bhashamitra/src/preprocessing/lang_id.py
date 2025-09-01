import re
from langdetect import detect
import langid

class HinglishLangID:
    def __init__(self):
        pass

    def detect_word_language(self, word: str) -> str:
        # Check Devanagari script
        if re.search("[\u0900-\u097F]", word):
            return "HIN"

        # Heuristic for common Hinglish words
        common_hindi = {"hain", "nahi", "kyun", "kal", "haan", "tum", "mera", "tera", "kya", "sab"}
        if word.lower() in common_hindi:
            return "HIN"

        # Try langid
        try:
            lang, _ = langid.classify(word)
            if lang == "en":
                return "ENG"
            elif lang == "hi":
                return "HIN"
        except:
            pass

        # Try langdetect fallback
        try:
            lang = detect(word)
            if lang == "en":
                return "ENG"
            elif lang == "hi":
                return "HIN"
        except:
            pass

        return "OTHER"

    def tag_sentence(self, sentence: str):
        words = sentence.split()
        return [(word, self.detect_word_language(word)) for word in words]

if __name__ == "__main__":
    detector = HinglishLangID()
    sample = "Haan let's meet kal at 5 pm"
    print(detector.tag_sentence(sample))
