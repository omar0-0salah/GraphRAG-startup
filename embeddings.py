
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


# ==========================================
# 1. Configuration
# ==========================================

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

VECTOR_INDEX_NAME = "company_description_embedding"


# ==========================================
# 2. Load embedding model once
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model():

    print("Loading embedding model...", flush=True)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded!", flush=True)

    return model


# ==========================================
# 3. Generate embeddings
# ==========================================

def generate_embedding(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot embed empty text.")

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError("Unexpected embedding dimensions.")

    return embedding.tolist()
# ==========================================
# Generate embedding for user question
# ==========================================

def embed_question(question):

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return generate_embedding(question)

# ==========================================
# 4. Prepare company information
# ==========================================

def build_company_document(company):

    name = company.get("name") or "Unknown company"

    details = [
        f"Company: {name}"
    ]

    if company.get("sector"):
        details.append(f"Sector: {company['sector']}")

    if company.get("headquarters"):
        details.append(
            f"Headquarters: {company['headquarters']}"
        )

    if company.get("founded") is not None:
        details.append(
            f"Founded: {company['founded']}"
        )

    if company.get("description"):
        details.append(
            f"Description: {company['description']}"
        )

    return "\n".join(details)


# ==========================================
# 5. Create Neo4j vector index
# ==========================================

def create_vector_index(driver, database):

    query = """
    CREATE VECTOR INDEX company_description_embedding IF NOT EXISTS
    FOR (c:Company)
    ON (c.description_embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }
    }
    """

    with driver.session(database=database) as session:
        session.run(query).consume()

    print("Vector index created or already exists.")


# ==========================================
# 6. Index company embeddings
# ==========================================

def index_company_embeddings(driver, database):

    create_vector_index(driver, database)

    fetch_query = """
    MATCH (c:Company)
    RETURN
        c.name AS name,
        c.sector AS sector,
        c.headquarters AS headquarters,
        c.founded AS founded,
        c.description AS description,
        c.embedding_source_text AS old_text,
        c.embedding_model AS old_model,
        c.description_embedding IS NOT NULL AS has_embedding
    ORDER BY c.name
    """

    update_query = """
    MATCH (c:Company {name: $name})
    SET
        c.description_embedding = $embedding,
        c.embedding_source_text = $source_text,
        c.embedding_model = $embedding_model,
        c.embedding_dimensions = $dimensions
    """

    with driver.session(database=database) as session:

        companies = session.run(fetch_query).data()

        print(f"Found {len(companies)} companies.")

        indexed = 0
        skipped = 0

        for company in companies:

            name = company["name"]

            document = build_company_document(company)

            # Skip only if the document and model are unchanged
            if (
                company["has_embedding"]
                and company["old_text"] == document
                and company["old_model"] == MODEL_NAME
            ):
                skipped += 1
                continue

            print(f"Embedding company: {name}")

            embedding = generate_embedding(document)

            session.run(
                update_query,
                name=name,
                embedding=embedding,
                source_text=document,
                embedding_model=MODEL_NAME,
                dimensions=EMBEDDING_DIMENSIONS
            ).consume()

            indexed += 1

    print("\nEmbedding process completed.")
    print(f"New or updated embeddings: {indexed}")
    print(f"Unchanged embeddings skipped: {skipped}")

    wait_for_vector_index(driver, database)


# ==========================================
# 7. Wait for Neo4j vector index
# ==========================================

def wait_for_vector_index(driver, database):

    query = """
    SHOW VECTOR INDEXES
    YIELD name, state
    WHERE name = $index_name
    RETURN state
    """

    for attempt in range(60):

        with driver.session(database=database) as session:

            record = session.run(
                query,
                index_name=VECTOR_INDEX_NAME
            ).single()

        if record is None:
            raise RuntimeError("Vector index not found.")

        state = record["state"]

        if state == "ONLINE":
            print("Vector index is ONLINE!")
            return

        if state == "FAILED":
            raise RuntimeError("Vector index creation failed.")

        print(f"Waiting for index... {state}")
        time.sleep(1)

    raise TimeoutError("Vector index did not become ONLINE.")


# ==========================================
# 8. Semantic search using question embedding
# ==========================================

def search_similar_companies(
    question,
    driver,
    database,
    top_k=5
):

    # Embed the user's question using the SAME model
    question_embedding = generate_embedding(question)

    query = """
    CALL db.index.vector.queryNodes(
        $index_name,
        $top_k,
        $embedding
    )
    YIELD node, score

    OPTIONAL MATCH (founder:Person)-[:FOUNDED]->(node)
    WITH node, score,
         collect(DISTINCT founder.name) AS founders

    OPTIONAL MATCH (investor:Investor)-[:INVESTED_IN]->(node)
    WITH node, score, founders,
         collect(DISTINCT investor.name) AS investors

    OPTIONAL MATCH (node)-[:DEVELOPS]->(product:Product)
    WITH node, score, founders, investors,
         collect(DISTINCT product.name) AS products

    OPTIONAL MATCH (node)-[:WON]->(award:Award)
    WITH node, score, founders, investors, products,
         collect(DISTINCT award.name) AS awards

    RETURN
        node.name AS company,
        node.sector AS sector,
        node.description AS description,
        founders,
        investors,
        products,
        awards,
        score AS similarity

    ORDER BY similarity DESC
    """

    with driver.session(database=database) as session:

        results = session.run(
            query,
            index_name=VECTOR_INDEX_NAME,
            top_k=top_k,
            embedding=question_embedding
        ).data()

    return results


# ==========================================
# 9. Run embedding indexing independently
# ==========================================

if __name__ == "__main__":

    URI = os.getenv("NEO4J_URI")
    USERNAME = os.getenv("NEO4J_USERNAME")
    PASSWORD = os.getenv("NEO4J_PASSWORD")
    DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([URI, USERNAME, PASSWORD]):
        raise ValueError("Check Neo4j configuration.")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        index_company_embeddings(driver, DATABASE)

    finally:
        driver.close()
