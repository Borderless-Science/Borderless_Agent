import sqlite3
import os
import glob
# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()
from textwrap import dedent

from IPython.display import display, Markdown
import pandas as pd
from pydantic import Field
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool, BaseTool
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# conn = sqlite3.connect("tcmbank_database.db")

# # Get all CSV files in a folder
# csv_files = glob.glob(os.path.join("..", "data", "raw", "tcm_bank_csv", "*.csv"))
# print(csv_files)

# # Loop through CSV files and import each into SQLite
# for file in csv_files:
#     # Use filename (without extension) as table name
#     table_name = file.split("\\")[-1].replace(".csv", "")

#     # Read CSV into DataFrame, trying 'latin-1' encoding
#     try:
#         df = pd.read_csv(file, encoding='latin-1')
#     except UnicodeDecodeError:
#         # If 'latin-1' fails, try another common encoding like 'cp1252'
#         try:
#             df = pd.read_csv(file, encoding='cp1252')
#         except Exception as e:
#             print(f"Could not read file {file} with latin-1 or cp1252 encoding: {e}")
#             continue # Skip to the next file

#     # Write to SQLite (if table exists, replace it)
#     df.to_sql(table_name, conn, if_exists="replace", index=False)

# print("All CSV files have been imported into SQLite!")
# conn.close()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
agent = LLM(model="gemini/gemini-2.5-flash-lite")

#db = SQLDatabase.from_uri("sqlite:///tcmbank_database.db")
db = SQLDatabase.from_uri("sqlite:///app/data/tcmbank_database.db")

@tool("list_tables")
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")


@tool("tables_schema")
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows for those
    tables. Be sure that the tables actually exist before calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)


@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)


@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Use this tool to double check if your query is correct before executing it. Always use this tool before
    using executing a query with `execute_sql`.
    """
    # Use the LangChain ChatGoogleGenerativeAI instance instead of the crewai.LLM wrapper
    return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})


search = GoogleSerperAPIWrapper


class SearchTool(BaseTool):
    name: str = "search"
    description: str = "A search engine. Useful for when you need to find information about current events or topics that are not in your database. Input should be a search query."
    search: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, query: str) -> str:
        """Execute the search query and return the results."""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


search_tool = SearchTool()

sql_dev = Agent(
    role="Senior Database Developer",
    goal="Construct and execute SQL queries based on a request",
    backstory=dedent(
        """
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        You have access to the following table descriptions to help you understand the database schema:

        *   **disease_all:** This table contains information about various diseases, including their names and potentially related Traditional Chinese Medicine (TCM) concepts. Key columns might include disease names in English and Chinese, and possibly links to other relevant data.<br>**Use For**: Disease identification, medical classifications, symptom mapping, disease relationships.
        *   **gene_all:** This table contains information about genes/proteins that serve as therapeutic targets with chromosomal locations and target validation status. Key columns might include gene identifiers and names.<br>**Use For**: Disease identification, medical classifications, symptom mapping, disease relationships.
        *   **herb_all:** This table contains information about traditional Chinese herbs. It includes details such as TCM names, English names, Latin names, properties, meridians, functions, and indications.<br>**Use For**: Traditional TCM knowledge, herb properties, meridian theory, classical indications
        *   **ingredient_all:** This table contains information about active chemical ingredients/compounds found in TCM herbs. It includes various identifiers, chemical properties, and potentially links to herbs or formulas.<br>**Use For**: Chemical structures, drug-likeness, pharmacokinetic properties, molecular identification

        ## Key Relationships & Data Flow

        **Primary Research Chain**: Herbs → Ingredients → Genes → Diseases

        ### Critical Connections:
        1. **TCMBank_ID**: Links herbs to their constituent ingredients
        2. **Source_ID**: Cross-references across external databases
        3. **Traditional-Modern Bridge**: Herb indications ↔ Disease classifications
        4. **Molecular Mechanisms**: Ingredient structures ↔ Gene targets ↔ Disease pathways

        ## Retrieval Instructions

        ### Query Processing Strategy:
        1. **Identify Query Type**:
          - Traditional TCM (herb names, properties, meridians)
          - Chemical/Molecular (compound names, structures, targets)
          - Medical (diseases, symptoms, conditions)
          - Mechanistic (how/why questions about TCM effects)

        2. **Multi-File Search Approach**:
          - **Single Entity Queries**: Start with the most relevant file, then expand
          - **Relationship Queries**: Search across multiple files simultaneously
          - **Mechanism Queries**: Follow the Herb→Ingredient→Gene→Disease pathway

        3. **Search Field Prioritization**:
          - **Primary**: Exact name matches, IDs
          - **Secondary**: Aliases, synonyms, alternative names
          - **Tertiary**: Descriptions, functions, classifications
          - **Contextual**: Related terms, broader categories

        ### Specific Retrieval Patterns:

        **For Traditional TCM Queries**:
        - Search herb_all for: TCM_name, Properties, Meridians, Function, Indication
        - Cross-reference with ingredient_all using TCMBank_ID
        - Link to disease_all through indication matching

        **For Chemical/Pharmacological Queries**:
        - Search ingredient_all for: compound names, molecular properties, ADMET data
        - Connect to gene_all for target information
        - Trace back to herb_all for source herbs

        **For Disease/Medical Queries**:
        - Search disease_all using multiple classification systems
        - Find related genes in gene_all
        - Identify targeting ingredients in ingredient_all
        - Trace to source herbs in herb_all

        **For Mechanism/Integration Queries**:
        - Follow complete pathway: specific herb → active ingredients → molecular targets → disease effects
        - Use cross-references and IDs to maintain data integrity
        - Combine traditional knowledge with molecular evidence


        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `check_sql` to check your queries for correctness.
        Use the `execute_sql` to execute queries against the database.
        Use the `search_tool` to find information outside of the database.
    """
    ),
    llm=agent,
    tools=[list_tables, tables_schema, execute_sql, check_sql, search_tool],
    allow_delegation=False,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal="Transform retrieved botanical research into structured JSON",
    backstory=dedent(
        """
        You have deep experience with analyzing datasets using Python.
        Your work is always based on the provided data and is clear,
        easy-to-understand and to the point. You have attention
        to detail and always produce very detailed work (as long as you need).
    """
    ),
    llm=agent,
    allow_delegation=False,
)

research_formatter = Agent(
    role="Formatter Specialist",
    goal="Transform retrieved botanical research into structured JSON",
    backstory=dedent(
        "You are a highly disciplined data formatter. "
        "Your sole responsibility is to take research output from the retriever agent "
        "and convert it into a concise, standardized JSON structure for UI rendering."
    ),
    instructions=dedent(
        """
    - Input: messy or verbose botanical research data from the retriever agent.
    - Output: valid JSON matching the schema below.
    - Do not include explanations, markdown, or extra text.
    - If information is missing, omit that field.
    - Always ensure valid JSON syntax.

    JSON Schema:
    {
      "specimen_description": {
        "botanical_name": "string",
        "common_names": ["string"],
        "part_used": "string",
        "preparation_form": "string",
        "morphology": "string"
      },
      "key_compounds": [
        {
          "compound": "string",
          "class": "string",
          "concentration_mg_g": number,
          "function": "string"
        }
      ],
      "compound_distribution": {
        "Flavonoid": number,
        "Phenolic acid": number,
        "Carotenoid": number,
        "Mineral": number
      },
      "toxicities_and_deficiencies": {
        "toxicities": ["string"],
        "deficiencies": ["string"]
      },
      "complementary_botanicals": {
        "iron_deficiency_anemia": ["string"],
        "enhanced_bioavailability": ["string"]
      },
      "treatable_ailments": ["string"],
      "pharmaceutical_comparison": [
        {
          "pharmaceutical": "string",
          "comparison": "string"
        }
      ]
    }

    botanical_name: Latin name of the herb.
    common_names: List of common names in English.
    part_used: Part of the plant used medicinally (e.g., leaves, roots).
    preparation_form: Form in which the herb is prepared (e.g., extract, powder).
    morphology: Description of the plant's physical characteristics.
    key_compounds: List of key chemical compounds found in the herb.
        compound: Name of the compound.
        class: Chemical class (e.g., Flavonoid, Alkaloid).
        concentration_mg_g: Concentration in mg/g.
        function: Biological function or effect.
    compound_distribution: Distribution of major compound classes as percentages.
        Flavonoid: Percentage of flavonoids.
        Phenolic acid: Percentage of phenolic acids.
        Carotenoid: Percentage of carotenoids.
        Mineral: Percentage of minerals.
    toxicities_and_deficiencies: Known toxicities and nutrient deficiencies.
        toxicities: List of known toxic effects.
        deficiencies: List of nutrient deficiencies caused by the herb.
    complementary_botanicals: Plants that enhance therapeutic effect or bioavailability when combined.
        iron_deficiency_anemia (example condition-specific synergy) → Plants that pair well to treat a given ailment.
        enhanced_bioavailability → Plants that improve absorption of key compounds.
    treatable_ailments: List of ailments treatable with this herb.
    pharmaceutical_comparison: Comparison with conventional pharmaceuticals.
    """
    ),
    llm=agent,
    allow_delegation=False,
)

# Create Tasks
extract_data = Task(
    description=("Extract data that is required for the query {query}."
                 " First figure out which tables to use and what SQL query to run."
                 "query the database using latin name if common name search fails."
                 " Then check the SQL query for correctness and execute it."
                 " Finally analyze the data and return the results."
                 " Only use the tools available to you. Do not make up any data."
                 "for each query, return: the botanical name, common names, part used, preparation form, key compounds, toxicities and deficiencies, complimentary botanicals, treatable ailments, pharmaceutical comparisons and morphology of the herb mentioned in the query."
                 " If you cannot find the data for any of these fields in the database, search for it using the search tool."
                 "If there are multiple herbs mentioned in the query, return the information for all of them. If no herbs are mentioned in the query, return an empty result."),
    expected_output="Database result for the query including botanical name, common names, part used, preparation form and morphology of the herb.",
    agent=sql_dev,
)

analyze_data = Task(
    description=("Analyze the data from the database and write an analysis for {query}."
                 " Make sure to base your analysis on the provided data and do not make up any information."
                 " If the data is incomplete or insufficient, state that in your analysis."
                 " Write a detailed analysis that covers all aspects of the data."
                 " The analysis should be easy to understand and to the point."
                 " Use bullet points, tables or other formatting to make the analysis clear."
                 " The analysis should be comprehensive and cover all relevant details."),
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data],
)

format_output = Task(
    description=(
        "Convert the retrieved information from the analysis into the standardized JSON format"
    ),
    expected_output="json format",
    agent=research_formatter,
    context=[analyze_data],
)

crew = Crew(
    agents=[sql_dev, data_analyst, research_formatter],
    tasks=[extract_data, analyze_data, format_output],
    process=Process.sequential,
    verbose=False,
    memory=False,
)

if __name__ == "__main__":
    inputs = {
        "query": "bitter leaf"
    }
    result = crew.kickoff(inputs=inputs)
    display(Markdown(result.raw))  