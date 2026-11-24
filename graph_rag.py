
import os
import json
import re
import traceback

from dotenv import load_dotenv
from neo4j import GraphDatabase
from groq import Groq

from retriever import retrieve_context

print("1. Loading environment...", flush=True)
load_dotenv()

# ----------------------------------
# Groq Configuration
# ----------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing!")

print("2. Creating Groq client...", flush=True)

llm_model = GROQ_MODEL

print("Using model:", llm_model, flush=True)

client = Groq(api_key=GROQ_API_KEY)

print("3. Groq client initialized successfully", flush=True)

# ----------------------------------
# Neo4j Configuration
# ----------------------------------

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not all([URI, USERNAME, PASSWORD]):
    raise ValueError("Neo4j configuration is incomplete!")

print("4. Initializing Neo4j...", flush=True)

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

print("5. Neo4j driver initialized", flush=True)


driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)



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

    print("Sending request to Groq...", flush=True)

    response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,
    max_tokens=1000
)

    output = response.choices[0].message.content.strip()

    print("Groq responded!", flush=True)
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

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

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