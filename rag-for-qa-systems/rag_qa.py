from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bedrock_client import invoke_claude


with open('tesla_knowledge_base.txt', 'r') as file:
    knowledge_base = [line.strip() for line in file if line.strip()]

print("Sample entries from the knowledge base:")
for entry in knowledge_base[:5]:
    print(entry)


vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
doc_vectors = vectorizer.fit_transform(knowledge_base)


def retrieve(query: str) -> str:
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)
    return knowledge_base[similarities.argmax()]


query = "When was Tesla founded?"
most_relevant_doc = retrieve(query)

print("\nQuery:", query)
print("Most Relevant Document:", most_relevant_doc)


print("\nBedrock client ready.")


def build_prompt(document: str, user_query: str) -> str:
    return (
        "You are an expert question answering assistant.\n\n"
        f"Use the following document to answer the query:\n\nDocument: {document}\n\nQuery: {user_query}"
    )


def generate_answer(document: str, user_query: str) -> None:
    try:
        result = invoke_claude(build_prompt(document, user_query), max_tokens=150)
        print("Generated Answer:", result["text"])
    except Exception as e:
        print(f"An error occurred while calling Claude via Bedrock for query '{user_query}': {e}")


generate_answer(most_relevant_doc, query)


queries = [
    "What is Tesla's mission?",
    "What is the Tesla Model S?",
    "Who founded Tesla?"
]

print()
for current_query in queries:
    print(f"Query: {current_query}")
    most_relevant_doc_for_query = retrieve(current_query)
    print(f"Retrieved Document: {most_relevant_doc_for_query}")
    generate_answer(most_relevant_doc_for_query, current_query)
    print("-" * 40)