
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

# Update this to your actual cleaned Excel file
file_path = r"C:\Users\Omar\OneDrive\Documents\GraphRAG-Project\data\tech_startups_dataset_cleaned.xlsx"

# Read relationship sheets
relationship_sheets = [
    "Founded",
    "WorksAt",
    "InvestedIn",
    "Acquired",
    "Develops",
    "Won"
]

relationships = {
    sheet: pd.read_excel(file_path, sheet_name=sheet)
    for sheet in relationship_sheets
}


# Convert pandas values into Python-compatible values
def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, str):
        value = value.strip()

    return value


def clean_record(record):
    return {
        key: clean_value(value)
        for key, value in record.items()
    }


# Import relationships

def import_relationships(tx, sheet, row):

    if sheet == "Founded":
        query = """
        MERGE (p:Person {name: $founder_name})
        MERGE (c:Company {name: $company_name})
        MERGE (p)-[r:FOUNDED]->(c)
        SET r.year = $year
        """

    elif sheet == "WorksAt":
        query = """
        MERGE (p:Person {name: $person_name})
        MERGE (c:Company {name: $company_name})
        MERGE (p)-[r:WORKS_AT {role: $role}]->(c)
        SET r.current = $current
        """

    elif sheet == "InvestedIn":
        query = """
        MERGE (i:Investor {name: $investor_name})
        MERGE (c:Company {name: $company_name})
        MERGE (i)-[r:INVESTED_IN {round: $round}]->(c)
        SET r.amount_million = $amount_million,
            r.year = $year
        """

    elif sheet == "Acquired":
        query = """
        MERGE (a:Company {name: $acquirer_company})
        MERGE (t:Company {name: $target_company})
        MERGE (a)-[r:ACQUIRED {year: $year}]->(t)
        SET r.amount_million = $amount_million
        """

    elif sheet == "Develops":
        query = """
        MERGE (c:Company {name: $company_name})
        MERGE (p:Product {name: $product_name})
        MERGE (c)-[:DEVELOPS]->(p)
        """

    elif sheet == "Won":
        if row["entity_type"].lower() == "company":
            query = """
            MERGE (e:Company {name: $entity_name})
            MERGE (a:Award {name: $award_name})
            MERGE (e)-[:WON]->(a)
            """
        else:
            query = """
            MERGE (e:Person {name: $entity_name})
            MERGE (a:Award {name: $award_name})
            MERGE (e)-[:WON]->(a)
            """

    else:
        raise ValueError(f"Unknown sheet: {sheet}")

    # Check required parameters before executing
    required_parameters = {
        "Founded": ["founder_name", "company_name", "year"],
        "WorksAt": ["person_name", "company_name", "role", "current"],
        "InvestedIn": ["investor_name", "company_name", "round", "amount_million", "year"],
        "Acquired": ["acquirer_company", "target_company", "year", "amount_million"],
        "Develops": ["company_name", "product_name"],
        "Won": ["entity_name", "entity_type", "award_name"]
    }

    missing = [
        key for key in required_parameters[sheet]
        if key not in row
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {sheet}: {missing}. "
            f"Available columns: {list(row.keys())}"
        )

    tx.run(query, **row)



# Execute imports
with driver.session(database=DATABASE) as session:

    for sheet, dataframe in relationships.items():

        count = 0

        for record in dataframe.to_dict("records"):

            row = clean_record(record)

            session.execute_write(
                import_relationships,
                sheet,
                row
            )

            count += 1

        print(f"{sheet}: imported {count} relationships")

driver.close()

print("Relationship import completed successfully!")
