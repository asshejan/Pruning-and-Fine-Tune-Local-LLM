import json
import random
import requests
from pathlib import Path

OPENSEARCH_URL = "http://localhost:9200"

def get_real_books(limit=1200):
    """Fetch real books from OpenSearch catalog."""
    query = {
        "size": limit,
        "query": {"term": {"is_enabled": True}},
        "_source": ["title", "title_bn", "authors", "authors_bn", "publisher", "price"]
    }
    r = requests.post(f"{OPENSEARCH_URL}/books_20260913114043/_search", json=query)
    hits = r.json().get("hits", {}).get("hits", [])
    books = []
    for h in hits:
        src = h["_source"]
        title = src.get("title") or src.get("title_bn")
        if not title:
            continue
        authors = src.get("authors") or []
        authors_bn = src.get("authors_bn") or []
        author = authors[0] if (authors and len(authors) > 0) else (authors_bn[0] if (authors_bn and len(authors_bn) > 0) else None)
        author_bn = authors_bn[0] if (authors_bn and len(authors_bn) > 0) else None
        books.append({
            "title": title,
            "title_bn": src.get("title_bn"),
            "author": author,
            "author_bn": author_bn,
            "price": src.get("price")
        })
    return books

# Realistic query patterns across Bangla, Banglish, and English
TEMPLATES = [
    # Banglish Price queries
    ("{title} er price koto?", "book_lookup", "price", True, False),
    ("{author} er {title} book er dam koto?", "book_lookup", "price", True, True),
    ("{title} koto taka?", "book_lookup", "price", True, False),
    
    # Bangla Price queries
    ("{title} বইয়ের দাম কত?", "book_lookup", "price", True, False),
    ("{author} এর {title} বইটির মূল্য কত?", "book_lookup", "price", True, True),
    
    # English Price queries
    ("How much is {title}?", "book_lookup", "price", True, False),
    ("What is the price of {title} by {author}?", "book_lookup", "price", True, True),
    
    # Availability queries
    ("{title} ache?", "book_lookup", "availability", True, False),
    ("{author} er {title} ki stock e ache?", "book_lookup", "availability", True, True),
    ("Is {title} available in stock?", "book_lookup", "availability", True, False),
    ("{title} বইটা কি স্টকে পাওয়া যাবে?", "book_lookup", "availability", True, False),
    
    # Author queries
    ("{title} car lekha?", "book_lookup", "author", True, False),
    ("Who is the writer of {title}?", "book_lookup", "author", True, False),
]

OFFTOPIC_QUERIES = [
    ("what is the weather today?", "decline", "offtopic"),
    ("cricket score update", "decline", "offtopic"),
    ("hello who are you?", "decline", "offtopic"),
    ("hi", "decline", "offtopic"),
]

VAGUE_QUERIES = [
    ("boi lagbe", "clarify", "vague"),
    ("humayun ahmed er boi", "clarify", "vague"),
    ("ekta golper boi dekhaw", "clarify", "vague"),
    ("suggest me a book", "clarify", "vague"),
]

def build_dataset():
    books = get_real_books(1500)
    samples = []

    for b in books:
        # Filter templates based on whether this book has an author
        available_templates = [t for t in TEMPLATES if not t[4] or (b["author"] or b["author_bn"])]
        tpl, branch, kind, has_title, has_author = random.choice(available_templates)
        
        # Pick English or Bangla variation
        use_bn = random.random() < 0.4 and b["title_bn"]
        title = b["title_bn"] if use_bn else b["title"]
        author = b["author_bn"] if (use_bn and b["author_bn"]) else (b["author"] or "")

        text = tpl.format(title=title, author=author)
        
        target = {
            "branch": branch,
            "kind": kind,
            "title": title if has_title else None,
            "author": author if has_author else None
        }
        samples.append({"query": text.strip(), "target": target})

    # Add offtopic & vague queries
    for q, br, kd in OFFTOPIC_QUERIES * 15:
        samples.append({"query": q, "target": {"branch": br, "kind": kd, "title": None, "author": None}})
    for q, br, kd in VAGUE_QUERIES * 15:
        samples.append({"query": q, "target": {"branch": br, "kind": kd, "title": None, "author": None}})

    random.shuffle(samples)

    # Format for Gemma Chat Template
    formatted = []
    for s in samples:
        formatted.append({
            "messages": [
                {"role": "user", "content": f"Route this user message into JSON: {s['query']}"},
                {"role": "model", "content": json.dumps(s["target"], ensure_ascii=False)}
            ]
        })

    # Split 90% train, 10% validation
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    with open("dataset_train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("dataset_val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_data)} train and {len(val_data)} validation samples.")

if __name__ == "__main__":
    build_dataset()
