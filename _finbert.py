from imports import *


def load_bert_model():
    """Load FOMC-specific FinBERT model"""
    
    model = BertForSequenceClassification.from_pretrained('ZiweiChen/FinBERT-FOMC', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('ZiweiChen/FinBERT-FOMC')
    
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def analyze_statement_sentiment(statement_text, sentiment_model):
    """Analyze single FOMC statement with FinBERT-FOMC - proper tokenization"""

    try:
        # Use the pipeline with proper truncation
        result = sentiment_model(statement_text, truncation=True, max_length=512)
        
        # FinBERT-FOMC outputs: Positive, Negative, Neutral (standard FinBERT format)
        raw_label = result[0]['label'].lower()
        sentiment_score = result[0]['score']
        
        return raw_label, sentiment_score
        
    except Exception as e:
        print(f"Error analyzing statement: {e}")
        return None, 0
