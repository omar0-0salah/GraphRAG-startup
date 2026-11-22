
import os
import json
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

from embeddings import embed_question


load_dotenv()

VECTOR_INDEX_NAME = "company_description_embedding"


# ==========================================
# 1. Semantic vector search
# ==========================================

def vector_search(question, driver, database, top_k=5):

    print("Generating question embedding...")

    question_embedding = embed_question(question)

    query = """
    CALL db.index.vector.queryNodes(
        $index_name,
        $top_k,
        $embedding
    )
    YIELD node, score

    RETURN
        node.name AS company,
        node.sector AS sector,
        node.description AS description,
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
# 2. Expand companies through graph relations
# ==========================================

def expand_company_graph(company_names, driver, database):

    if not company_names:
        return []

    query = """
    UNWIND $company_names AS company_name

    MATCH (c:Company {name: company_name})

    CALL {
        WITH c
        OPTIONAL MATCH (f:Person)-[:FOUNDED]->(c)
        RETURN collect(DISTINCT f.name) AS founders
    }

    CALL {
        WITH c
        OPTIONAL MATCH (i:Investor)-[r:INVESTED_IN]->(c)
        RETURN collect(DISTINCT
            CASE WHEN i IS NULL THEN null
            ELSE {
                name: i.name,
                round: r.round,
                amount_million: r.amount_million,
                year: r.year
            } END
        ) AS investors
    }

    CALL {
        WITH c
        OPTIONAL MATCH (c)-[:DEVELOPS]->(p:Product)
        RETURN collect(DISTINCT p.name) AS products
    }

    CALL {
        WITH c
        OPTIONAL MATCH (c)-[:WON]->(a:Award)
        RETURN collect(DISTINCT a.name) AS awards
    }

    CALL {
        WITH c
        OPTIONAL MATCH (person:Person)-[w:WORKS_AT]->(c)
        RETURN collect(DISTINCT
            CASE WHEN person IS NULL THEN null
            ELSE {
                name: person.name,
                role: w.role,
                current: w.current
            } END
        ) AS employees
    }

    CALL {
        WITH c
        OPTIONAL MATCH (c)-[r:ACQUIRED]->(target:Company)
        RETURN collect(DISTINCT
            CASE WHEN target IS NULL THEN null
            ELSE {
                name: target.name,
                year: r.year,
                amount_million: r.amount_million
            } END
        ) AS acquired_companies
    }

    CALL {
        WITH c
        OPTIONAL MATCH (acquirer:Company)-[r:ACQUIRED]->(c)
        RETURN collect(DISTINCT
            CASE WHEN acquirer IS NULL THEN null
            ELSE {
                name: acquirer.name,
                year: r.year,
                amount_million: r.amount_million
            } END
        ) AS acquired_by
    }

    RETURN
        c.name AS company,
        c.sector AS sector,
        c.description AS description,
        founders,
        investors,
        products,
        awards,
        employees,
        acquired_companies,
        acquired_by
    """

    with driver.session(database=database) as session:

        results = session.run(
            query,
            company_names=company_names
        ).data()

    return results


# ==========================================
# 3. Exact entity matching
# ==========================================

def find_entity_paths(question, driver, database):

    query = """
    MATCH (start)

    WHERE start.name IS NOT NULL
      AND size(start.name) > 1
      AND toLower($question) CONTAINS toLower(start.name)

    MATCH path = (start)-[*1..2]-(other)

    RETURN DISTINCT
        labels(start) AS matched_labels,
        start.name AS matched_name,

        [n IN nodes(path) |
            {
                labels: labels(n),
                name: n.name
            }
        ] AS path_nodes,

        [r IN relationships(path) |
            {
                type: type(r),
                properties: properties(r)
            }
        ] AS path_relationships

    LIMIT 100
    """

    with driver.session(database=database) as session:

        results = session.run(
            query,
            question=question
        ).data()

    return results


# ==========================================
# 4. Detect general founder questions
# ==========================================

def is_global_founder_question(question):

    q = question.lower()

    has_founder_intent = bool(
        re.search(r"\b(founder|founders|founded)\b", q)
    )

    general_terms = [
        "who are",
        "all",
        "every",
        "list",
        "show all"
    ]

    is_general = any(
        term in q for term in general_terms
    )

    return has_founder_intent and is_general


# ==========================================
# 5. Direct graph retrieval
# ==========================================

def retrieve_all_founders(driver, database):

    query = """
    MATCH (p:Person)-[:FOUNDED]->(c:Company)

    RETURN
        p.name AS founder,
        collect(c.name) AS companies

    ORDER BY founder
    """

    with driver.session(database=database) as session:

        results = session.run(query).data()

    return results


# ==========================================
# 6. Main retrieval router
# ==========================================



def retrieve_context(question, driver, database, top_k=3):

    # Handle general questions about all founders
    if is_global_founder_question(question):
        results = retrieve_all_founders(driver, database)

        return {
            "type": "global_graph_search",
            "results": results
        }

    # 1. Search for similar companies using embeddings
    vector_results = vector_search(
        question, driver, database, top_k=top_k
    )

    # 2. Extract company names
    company_names = [
        item["company"] for item in vector_results
    ]

    # 3. Retrieve their graph relationships
    graph_results = expand_company_graph(
        company_names, driver, database
    )

    # 4. Keep similarity scores from vector search
    similarity_map = {
        item["company"]: item["similarity"]
        for item in vector_results
    }

    # 5. Build simplified results for Gemini
    compact_results = []

    for item in graph_results:
        compact_results.append({
            "company": item["company"],
            "sector": item["sector"],
            "description": item["description"],
            "founders": item["founders"],
            "investors": [
                investor["name"]
                for investor in item["investors"]
                if investor
            ],
            "products": item["products"],
            "awards": item["awards"],
            "employees": item["employees"],
            "acquired_companies": item["acquired_companies"],
            "acquired_by": item["acquired_by"],
            "similarity": round(
                similarity_map.get(item["company"], 0), 3
            )
        })

    return {
        "type": "semantic_graph_search",
        "results": compact_results
    }



# ==========================================
# 7. Test retriever independently
# ==========================================

if __name__ == "__main__":

    URI = os.getenv("NEO4J_URI")
    USERNAME = os.getenv("NEO4J_USERNAME")
    PASSWORD = os.getenv("NEO4J_PASSWORD")
    DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:

        question = input(
            "\nAsk a question about the startup graph: "
        )

        results = retrieve_context(
            question,
            driver,
            DATABASE,
            top_k=3
        )

        print("\nRetrieval type:")
        print(results["type"])

        print("\nRetrieved information:")

        print(
            json.dumps(
                results,
                indent=4,
                ensure_ascii=False,
                default=str
            )
        )

    except Exception:
        import traceback
        traceback.print_exc()

    finally:
        driver.close()
