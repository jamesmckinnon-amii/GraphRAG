import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
import google.generativeai as genai

class BuildingCodeLoader:
    def __init__(self, uri, user, password, google_api_key=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.google_api_key = google_api_key
        else:
            self.google_api_key = None
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Delete all nodes and relationships"""
        with self.driver.session() as session:
            print("Clearing database...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared!")
    
    def setup_database(self):
        """Create basic constraints and indexes"""
        with self.driver.session() as session:
            # Unique constraint on section ID
            session.run("""
                CREATE CONSTRAINT section_id IF NOT EXISTS
                FOR (s:Section) REQUIRE s.id IS UNIQUE
            """)
            
            # Unique constraint on table ID
            session.run("""
                CREATE CONSTRAINT table_id IF NOT EXISTS
                FOR (t:Table) REQUIRE t.id IS UNIQUE
            """)
            
            # Full-text search index for sections
            session.run("""
                CREATE FULLTEXT INDEX section_search IF NOT EXISTS
                FOR (s:Section) ON EACH [s.text, s.title]
            """)
            
            # Full-text search index for tables
            session.run("""
                CREATE FULLTEXT INDEX table_search IF NOT EXISTS
                FOR (t:Table) ON EACH [t.name, t.content]
            """)
    
    def load_tables(self, section_id, tables_dict, include_ancestors=False):
        """Load tables for a section and create relationships
        
        Args:
            section_id: The section containing these tables
            tables_dict: Dictionary of tables
            include_ancestors: If True, creates IS_IN relationships to all ancestor sections
                             If False, only creates relationship to immediate parent section
        """
        with self.driver.session() as session:
            for table_id, table_data in tables_dict.items():
                # Create the table node
                session.run("""
                    MERGE (t:Table {id: $id})
                    SET t.name = $name,
                        t.content = $content
                """, 
                    id=table_id,
                    name=table_data.get("table_name", ""),
                    content=table_data.get("table_content", "")
                )
                
                # Create bidirectional relationships between table and section
                session.run("""
                    MATCH (t:Table {id: $table_id})
                    MATCH (s:Section {id: $section_id})
                    MERGE (t)-[:IS_IN_SECTION]->(s)
                    MERGE (s)-[:HAS_TABLE]->(t)
                """, table_id=table_id, section_id=section_id)
                
                # Optionally create relationships to all ancestor sections
                if include_ancestors:
                    session.run("""
                        MATCH (t:Table {id: $table_id})
                        MATCH (s:Section {id: $section_id})
                        MATCH path = (s)-[:HAS_SUBSECTION*]->(ancestor:Section)
                        WITH t, ancestor
                        MERGE (t)-[:IS_IN_ANCESTOR]->(ancestor)
                    """, table_id=table_id, section_id=section_id)
    
    def load_section(self, section_id, section_data, parent_id=None):
        """Load a section, its tables, and its subsections recursively"""
        with self.driver.session() as session:
            # Create the section node
            session.run("""
                MERGE (s:Section {id: $id})
                SET s.title = $title,
                    s.text = $text
            """, 
                id=section_id,
                title=section_data.get("title", ""),
                text=section_data.get("text", "")
            )
            
            # Link to parent if exists
            if parent_id:
                session.run("""
                    MATCH (parent:Section {id: $parent_id})
                    MATCH (child:Section {id: $child_id})
                    MERGE (parent)-[:HAS_SUBSECTION]->(child)
                """, parent_id=parent_id, child_id=section_id)
            
            # Load tables for this section
            tables = section_data.get("tables", {})
            if tables:
                self.load_tables(section_id, tables)
            
            # Recursively load subsections
            for sub_id, sub_data in section_data.get("subsections", {}).items():
                self.load_section(sub_id, sub_data, section_id)
    
    def create_reference_relationships(self, section_id, referenced_items):
        """Create REFERENCES relationships from a section to other sections and tables it references
        
        Args:
            section_id: The ID of the section that contains references
            referenced_items: List of section IDs or table IDs that are referenced
                            (e.g., ["9.23.3.4.", "Table 9.23.3.4.", "9.20.1.1."])
        """
        if not referenced_items:
            return
        
        with self.driver.session() as session:
            for ref_id in referenced_items:
                # Check if this is a table reference (starts with "Table ")
                if ref_id.startswith("Table "):
                    # Create relationship to a Table node
                    result = session.run("""
                        MATCH (source:Section {id: $source_id})
                        MATCH (target:Table {id: $ref_id})
                        MERGE (source)-[:REFERENCES]->(target)
                        RETURN source.id as source, target.id as target
                    """, source_id=section_id, ref_id=ref_id)
                    
                    # Check if the relationship was created
                    if result.single():
                        pass  # Successfully created
                    else:
                        print(f"    Warning: Table '{ref_id}' referenced by {section_id} not found in graph")
                else:
                    # Create relationship to a Section node
                    result = session.run("""
                        MATCH (source:Section {id: $source_id})
                        MATCH (target:Section {id: $ref_id})
                        MERGE (source)-[:REFERENCES]->(target)
                        RETURN source.id as source, target.id as target
                    """, source_id=section_id, ref_id=ref_id)
                    
                    # Check if the relationship was created
                    if result.single():
                        pass  # Successfully created
                    else:
                        print(f"    Warning: Section '{ref_id}' referenced by {section_id} not found in graph")
            
    
    def process_references_recursive(self, section_id, section_data):
        """Process referenced_text for a section and all its subsections recursively
        
        Args:
            section_id: The section ID
            section_data: The section data dictionary
        """
        # Process references for current section
        referenced_sections = section_data.get("referenced_text", [])
        if referenced_sections:
            self.create_reference_relationships(section_id, referenced_sections)
        
        # Recursively process subsections
        for sub_id, sub_data in section_data.get("subsections", {}).items():
            self.process_references_recursive(sub_id, sub_data)
    
    def load_all_references(self, code_dict):
        """Load all REFERENCES relationships for the entire building code
        
        Args:
            code_dict: The building code dictionary
        """
        print("Creating REFERENCES relationships...")
        for section_id, section_data in code_dict.items():
            print(f"  Processing references for {section_id}")
            self.process_references_recursive(section_id, section_data)
        print("References loaded!")
    
    def is_article_section(self, section_id):
        """Check if a section ID represents an Article (4+ number levels)
        
        Examples: 9.4.1.1., 9.4.1.2.
        """
        # Count the number of numeric parts
        parts = section_id.strip('.').split('.')
        return len(parts) == 4
    
    def generate_embedding(self, text, model="models/text-embedding-004"):
        """Generate embedding for text using Google's Gemini embedding model
        
        Args:
            text: Text to embed
            model: Google embedding model to use
                   - "models/text-embedding-004" (768 dimensions, latest)
                   - "models/embedding-001" (768 dimensions, older)
        
        Returns:
            List of floats representing the embedding vector
        """
        if not self.google_api_key:
            raise ValueError("Google API key not configured")
        
        # Google's text-embedding-004 supports up to 2048 tokens
        # Truncate if needed (rough estimate: 1 token ≈ 4 characters)
        if len(text) > 8000:
            text = text[:8000]
        
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"  # Optimized for retrieval tasks
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def create_text_for_embedding(self, section_id, section_data):
        """Create a comprehensive text representation for embedding
        
        Combines title, text, and context for better semantic search
        """
        parts = []
        
        # Add section ID as context
        parts.append(f"Section {section_id}")
        
        # Add title
        if section_data.get("title"):
            parts.append(f"Title: {section_data['title']}")
        
        # Add main text
        if section_data.get("text"):
            parts.append(section_data["text"])
        
        return "\n\n".join(parts)
    
    def add_embeddings_to_articles(self, code_dict):
        """Generate and store embeddings for all Article-level sections
        
        Args:
            code_dict: The building code dictionary
        """
        if not self.google_api_key:
            print("Warning: No Google API key configured. Skipping embeddings.")
            return
        
        print("Generating embeddings for Article sections...")
        
        articles_to_process = []
        
        # Collect all article sections recursively
        def collect_articles(section_id, section_data):
            if self.is_article_section(section_id):
                # Only add if section has meaningful text
                text = section_data.get("text", "").strip()
                if text and len(text) > 10:  # Avoid empty or trivial sections
                    articles_to_process.append((section_id, section_data))
            
            # Recurse into subsections
            for sub_id, sub_data in section_data.get("subsections", {}).items():
                collect_articles(sub_id, sub_data)
        
        # Collect all articles
        for section_id, section_data in code_dict.items():
            collect_articles(section_id, section_data)
        
        print(f"Found {len(articles_to_process)} articles to embed")
        
        # Process each article
        for i, (section_id, section_data) in enumerate(articles_to_process):
            try:
                # Create text for embedding
                text = self.create_text_for_embedding(section_id, section_data)
                
                # Generate embedding using Gemini
                embedding = self.generate_embedding(text)
                
                # Store in Neo4j
                with self.driver.session() as session:
                    session.run("""
                        MATCH (s:Section {id: $section_id})
                        SET s.embedding = $embedding,
                            s.embedding_text = $text
                    """, section_id=section_id, embedding=embedding, text=text)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(articles_to_process)} articles")
                
            except Exception as e:
                print(f"  Error processing {section_id}: {e}")
                continue
        
        print("Embeddings generated")
    
    def setup_vector_index(self):
        """Create a vector index for similarity search on Article sections"""
        with self.driver.session() as session:
            try:
                # Drop existing index if it exists
                session.run("DROP INDEX article_embeddings IF EXISTS")
                
                # Create vector index
                session.run("""
                    CREATE VECTOR INDEX article_embeddings IF NOT EXISTS
                    FOR (s:Section) ON (s.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("Vector index created!")
            except Exception as e:
                print(f"Error creating vector index: {e}")
    
    def load_all(self, code_dict, clear_first=False, include_embeddings=False):
        """Load entire building code
        
        Args:
            code_dict: The building code dictionary
            clear_first: If True, clears the database before loading
            include_embeddings: If True, generates embeddings for Articles
        """
        if clear_first:
            self.clear_database()
        
        print("Setting up database...")
        self.setup_database()
        
        print("Loading sections and tables...")
        for section_id, section_data in code_dict.items():
            print(f"  Loading {section_id}")
            self.load_section(section_id, section_data)
        
        print("\nLoading references...")
        self.load_all_references(code_dict)
        
        if include_embeddings:
            print("\nGenerating embeddings...")
            self.add_embeddings_to_articles(code_dict)
            print("\nSetting up vector index...")
            self.setup_vector_index()
        
        print("\nDone!")

# Example usage
if __name__ == "__main__":
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    with open('../data/processed/Part_9_National_building_Code.json', 'r') as f:
        building_code = json.load(f)
    
    # Connect and load
    loader = BuildingCodeLoader(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        google_api_key=GOOGLE_API_KEY
    )
    
    try:
        loader.load_all(
            building_code, 
            clear_first=True,
            include_embeddings=True  # Set to True to generate embeddings
        )
    finally:
        loader.close()