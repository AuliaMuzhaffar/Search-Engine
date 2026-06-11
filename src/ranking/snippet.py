import re
from typing import List

class SnippetGenerator:
    @staticmethod
    def generate(text: str, query: str, length: int = 160) -> str:
        """
        Generates a text snippet around the first occurrence of query terms.
        """
        if not text:
            return ""
        if not query:
            return text[:length] + "..." if len(text) > length else text

        # Clean multiple spaces and newlines
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Split query into terms
        query_terms = [re.escape(term.lower()) for term in query.split() if len(term) > 1]
        
        if not query_terms:
            return clean_text[:length] + "..." if len(clean_text) > length else clean_text

        # Find first occurrence of any query term
        pattern = "|".join(query_terms)
        match = re.search(pattern, clean_text, re.IGNORECASE)
        
        if not match:
            # Fallback to start of text
            return clean_text[:length] + "..." if len(clean_text) > length else clean_text

        start_idx = match.start()
        
        # Calculate window around match
        half_len = length // 2
        start = max(0, start_idx - half_len)
        end = min(len(clean_text), start_idx + half_len)
        
        # Adjust start/end to avoid splitting words
        if start > 0:
            first_space = clean_text.find(' ', start, start_idx)
            if first_space != -1:
                start = first_space + 1
                
        if end < len(clean_text):
            last_space = clean_text.rfind(' ', start_idx, end)
            if last_space != -1 and last_space > start_idx:
                end = last_space

        snippet = clean_text[start:end].strip()
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(clean_text):
            snippet = snippet + "..."
            
        return snippet
