
from bs4 import BeautifulSoup
import unicodedata
import re
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

def strip_html(text):
    
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text()
    
def normalize_unicode(text):
    
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)

def clean_whitespace(text):
    
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_english(text):
   
    if not text or not text.strip():
        return False
    try:
        return detect(text[:1000]) == "en"
    except LangDetectException:
        return False

def clean_text(text):
  
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    if not text or not is_english(text):
        return None
    return text