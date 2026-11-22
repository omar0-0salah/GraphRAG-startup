import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Excel file
FILE_PATH = BASE_DIR / "data" / "tech_startups_dataset_cleaned.xlsx"

# Connect to Neo4j
driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

driver.verify_connectivity()
print("Connected to Neo4j successfully!")


def clean_properties(row):
    """Remove null properties and convert integral floats to integers."""
    properties = {}

    for key, value in row.items():
        if pd.isna(value):
            continue

        if isinstance(value, float) and value.is_integer():
            value = int(value)

        properties[key] = value

    return properties


def create_nodes(tx, label, properties):
    query = f"""
    MERGE (n:{label} {{name: $name}})
    SET n += $properties
    """

    tx.run(
        query,
        name=properties["name"],
        properties=properties
    )


def import_nodes():
    # Read node sheets
    founders = pd.read_excel(FILE_PATH, sheet_name="Founders")
    companies = pd.read_excel(FILE_PATH, sheet_name="Companies")
    investors = pd.read_excel(FILE_PATH, sheet_name="Investors")
    products = pd.read_excel(FILE_PATH, sheet_name="Products")
    awards = pd.read_excel(FILE_PATH, sheet_name="Awards")

    with driver.session(database=DATABASE) as session:

        # Person nodes
        for _, row in founders.iterrows():
            properties = clean_properties(row)
            session.execute_write(
                create_nodes, "Person", properties
            )

        # Also create people appearing in WorksAt
        works_at = pd.read_excel(FILE_PATH, sheet_name="WorksAt")

        for name in works_at["person_name"].dropna().unique():
            session.execute_write(
                create_nodes,
                "Person",
                {"name": str(name).strip()}
            )

        # Company nodes
        for _, row in companies.iterrows():
            properties = clean_properties(row)
            session.execute_write(
                create_nodes, "Company", properties
            )

        # Investor nodes
        for _, row in investors.iterrows():
            properties = clean_properties(row)
            session.execute_write(
                create_nodes, "Investor", properties
            )

        # Product nodes
        for _, row in products.iterrows():
            properties = clean_properties(row)
            session.execute_write(
                create_nodes, "Product", properties
            )

        # Award nodes
        for _, row in awards.iterrows():
            properties = clean_properties(row)
            session.execute_write(
                create_nodes, "Award", properties
            )

    print("All nodes imported successfully!")


try:
    import_nodes()
finally:
    driver.close()