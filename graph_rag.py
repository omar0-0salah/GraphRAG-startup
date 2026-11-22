
import os
import json
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase
from google import genai
import traceback
from retriever import retrieve_context

print("1. Loading environment...", flush=True)
load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

print("2. Creating Gemini client...", flush=True)
client = genai.Client(api_key=GEMINI_API_KEY)

# Neo4j configuration
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("5. Initializing Neo4j...", flush=True)

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

print("6. Neo4j initialized!", flush=True)

print("Both clients initialized successfully!", flush=True)
client = genai.Client(api_key=GEMINI_API_KEY)


# 1. Define the graph schema
GRAPH_SCHEMA = """
Nodes:

Person:
- name
- born
- role
- hometown

Company:
- name
- founded
- sector
- headquarters
- valuation_billion
- description

Investor:
- name
- founded
- type
- hq

Product:
- name
- category
- launch_year

Award:
- name
- category
- year


Relationships:

(Person)-[:FOUNDED {year}]->(Company)

(Person)-[:WORKS_AT {role, current}]->(Company)

(Investor)-[:INVESTED_IN {
    round,
    amount_million,
    year
}]->(Company)

(Company)-[:ACQUIRED {
    year,
    amount_million
}]->(Company)

(Company)-[:DEVELOPS]->(Product)

(Person or Company)-[:WON]->(Award)
"""


# 2. Generate Cypher from a natural-language question
def generate_cypher(question):

    print("Generating Cypher...", flush=True)

    prompt = f"""
    You are a Neo4j Cypher expert.

    Database schema:
    Person(name, born, role, hometown)
    Company(name, founded, sector, headquarters)
    Investor(name, founded, type, hq)
    Product(name, category, launch_year)
    Award(name, category, year)

    Relationships:
    Person-[:FOUNDED]->Company
    Person-[:WORKS_AT]->Company
    Investor-[:INVESTED_IN]->Company
    Company-[:ACQUIRED]->Company
    Company-[:DEVELOPS]->Product
    Company-[:WON]->Award
    Person-[:WON]->Award

    Return ONLY JSON:
    {{
        "cypher": "MATCH ... RETURN ...",
        "parameters": {{}}
    }}

    User question: {question}
    """

    print("Sending request to Gemini...", flush=True)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.1}
    )

    print("Gemini responded!", flush=True)
    print(response.text, flush=True)

    output = response.text.strip()
    output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output).strip()

    result = json.loads(output)

    return result["cypher"], result["parameters"]


# 3. Validate generated Cypher
def validate_cypher(query):

    if not query or not query.strip():
        raise ValueError("The generated query is empty.")

    normalized = query.strip()

    # Only allow queries beginning with MATCH
    if not re.match(
        r"^(MATCH|OPTIONAL\s+MATCH)\b",
        normalized,
        re.IGNORECASE
    ):
        raise ValueError("Only MATCH-based queries are allowed.")

    # Reject multiple statements
    if ";" in normalized:
        raise ValueError("Multiple Cypher statements are not allowed.")

    # Block write operations and unrestricted procedures
    forbidden = [
        "CREATE", "MERGE", "DELETE", "DETACH",
        "SET", "REMOVE", "DROP", "CALL",
        "LOAD CSV", "FOREACH", "UNION"
    ]

    for keyword in forbidden:
        pattern = r"\b" + re.escape(keyword) + r"\b"

        if re.search(pattern, normalized, re.IGNORECASE):
            raise ValueError(
                f"Forbidden Cypher operation: {keyword}"
            )

    if not re.search(r"\bRETURN\b", normalized, re.IGNORECASE):
        raise ValueError("Query must contain RETURN.")


# 4. Execute graph retrieval
def retrieve_from_graph(question):
    print("1. Generating Cypher...")

    query, parameters = generate_cypher(question)

    print("2. Generated Cypher:")
    print(query)
    print("Parameters:", parameters)

    print("3. Validating Cypher...")
    validate_cypher(query)

    print("4. Connecting to Neo4j...")

    with driver.session(database=DATABASE) as session:
        results = session.execute_read(
            lambda tx: [
                record.data()
                for record in tx.run(query, parameters)
            ]
        )

    print("5. Query executed successfully!")
    return query,results




def generate_answer(question, retrieval_results):

    # Extract ONLY the retrieved results
    context = json.dumps(
        retrieval_results["results"],
        ensure_ascii=False,
        default=str
    )

    if not retrieval_results["results"]:
        return "No relevant information was found in the knowledge graph."

    prompt = f"""
    Answer the user's question using ONLY the information below.
    Do not invent facts.
    If the information is insufficient, say so.
    Keep your answer clear and concise.

    Question:
    {question}

    Retrieved information:
    {context}
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 500
        }
    )

    return response.text.strip()



# ==========================================
# Run complete GraphRAG pipeline
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nAsk a question about the startup graph: "
    )

    try:

        # 1. Retrieve information without Gemini
        print("\n1. Retrieving information...")

        retrieval_results = retrieve_context(
            question=question,
            driver=driver,
            database=DATABASE,
            top_k=3
        )

        # 2. Display retrieval strategy
        print("\n2. Retrieval type:")
        print(retrieval_results["type"])

        # 3. Display retrieved evidence
        print("\n3. Retrieved information:")

        print(
            json.dumps(
                retrieval_results,
                indent=4,
                ensure_ascii=False,
                default=str
            )
        )

        # 4. Generate final answer using Gemini
        print("\n4. Generating final answer...")

        answer = generate_answer(
            question,
            retrieval_results
        )

        # 5. Display answer
        print("\n========== GraphRAG Answer ==========")
        print(answer)
        print("=====================================")

    except Exception:
        traceback.print_exc()

    finally:
        driver.close()