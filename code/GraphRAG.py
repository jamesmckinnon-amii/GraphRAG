import os
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class BuildingCodeRAG:
    def __init__(self, uri, user, password, google_api_key):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
    
    def close(self):
        self.driver.close()
    
    def generate_query_embedding(self, query, model="models/text-embedding-004"):
        """Generate embedding for a query using Google's Gemini"""
        result = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query" 
        )
        return result['embedding']
    
    def find_similar_articles(self, query, top_k=5):
        """Find most similar articles using vector search"""
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Vector similarity search
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('article_embeddings', $top_k, $query_embedding)
                YIELD node, score
                RETURN node.id as section_id, 
                       node.title as title, 
                       node.text as text,
                       score
                ORDER BY score DESC
            """, query_embedding=query_embedding, top_k=top_k)
            
            return [dict(record) for record in result]
    
    def get_context_for_section(self, section_id):
        """Get rich context around a section by traversing the graph
        
        Returns:
            - Parent sections (for broader context)
            - Referenced sections (for dependencies)
            - Referenced tables (tables directly referenced by this section)
            - Associated tables (tables owned by this section)
            - Child sections
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {id: $section_id})
                
                // Get parent chain
                OPTIONAL MATCH parent_path = (s)<-[:HAS_SUBSECTION*1..]-(parent:Section)
                WITH s, parent_path, parent
                ORDER BY length(parent_path) DESC
                WITH s, collect(DISTINCT {
                    id: parent.id, 
                    title: parent.title,
                    text: parent.text,
                    level: length(parent_path)
                }) as parents
                
                // Get referenced sections and their tables
                OPTIONAL MATCH (s)-[:REFERENCES]->(ref:Section)
                OPTIONAL MATCH (ref)-[:HAS_TABLE]->(ref_table:Table)
                WITH s, parents, ref, collect(DISTINCT {
                    id: ref_table.id,
                    name: ref_table.name,
                    content: ref_table.content
                }) as ref_tables
                WITH s, parents, collect(DISTINCT {
                    id: ref.id,
                    title: ref.title,
                    text: ref.text,
                    tables: CASE WHEN ref_tables[0].id IS NOT NULL THEN ref_tables ELSE [] END
                }) as references
                
                // Get tables directly associated with this section (owned by this section)
                OPTIONAL MATCH (s)-[:HAS_TABLE]->(t:Table)
                WITH s, parents, references, collect(DISTINCT {
                    id: t.id,
                    name: t.name,
                    content: t.content
                }) as tables
                
                // Get tables that are directly referenced by this section
                // These are tables mentioned in the text but may belong to other sections
                OPTIONAL MATCH (s)-[:REFERENCES]->(ref_table:Table)
                WITH s, parents, references, tables, collect(DISTINCT {
                    id: ref_table.id,
                    name: ref_table.name,
                    content: ref_table.content
                }) as referenced_tables
                
                RETURN s.id as section_id,
                    s.title as title,
                    s.text as text,
                    parents,
                    references,
                    tables,
                    referenced_tables
            """, section_id=section_id)
            
            return result.single()
    
    def format_table_for_prompt(self, table):
        """Format a table for inclusion in the prompt"""
        parts = []
        # Always include the table ID first for clear identification
        if table.get('id'):
            parts.append(f"{table['id']}")
        # Then add the table name/title if it exists
        if table.get('name'):
            parts.append(f"{table['name']}")
        # Finally add the table content
        if table.get('content'):
            parts.append(table['content'])
        return "\n".join(parts)
    
    def answer_question(self, question, top_k=3, max_context_length=15000):
        """Full RAG pipeline: vector search + graph traversal + LLM generation
        
        Args:
            question: The user's question
            top_k: Number of similar articles to retrieve
            max_context_length: Maximum character length for context (to avoid token limits)
        """
        
        # Find relevant starting points
        print(f"Searching for relevant sections...")
        similar_articles = self.find_similar_articles(question, top_k=top_k)
        
        if not similar_articles:
            return {
                "answer": "I couldn't find any relevant sections in the building code for this question.",
                "source_sections": [],
                "contexts": []
            }
        
        # Get context for each article
        print(f"Gathering context from knowledge graph...")
        contexts = []
        for article in similar_articles:
            context = self.get_context_for_section(article['section_id'])
            if context:
                contexts.append(context)
        
        # Build prompt with structured context
        prompt_parts = [
            "You are an expert on the National Building Code. Answer the following question based on the provided sections.",
            f"\nQuestion: {question}\n",
            "\nRelevant Building Code Sections:\n"
        ]
        
        current_length = len("\n".join(prompt_parts))
        
        for i, ctx in enumerate(contexts, 1):
            section_parts = []
            section_parts.append(f"\n{'='*60}")
            section_parts.append(f"Section {i}: {ctx['section_id']}")
            section_parts.append(f"{'='*60}")
            section_parts.append(f"Title: {ctx['title']}")
            
            # Add parent context for hierarchy
            if ctx['parents']:
                parent_titles = " > ".join([p['title'] for p in reversed(ctx['parents'])])
                section_parts.append(f"Context: {parent_titles}")
            
            # Add main section content
            section_parts.append(f"\nContent:")
            section_parts.append(ctx['text'])
            
            # Add tables directly associated with this section
            if ctx['tables'] and any(t.get('id') for t in ctx['tables']):
                section_parts.append(f"\n--- Tables for Section {ctx['section_id']} ---")
                for table in ctx['tables']:
                    if table.get('id'):  # Only include if table exists
                        section_parts.append(f"\n{self.format_table_for_prompt(table)}")
            
            # Add referenced tables (NEW: tables mentioned in text but owned by other sections)
            if ctx.get('referenced_tables') and any(t.get('id') for t in ctx['referenced_tables']):
                section_parts.append(f"\n--- Referenced Tables (from other sections) ---")
                for table in ctx['referenced_tables']:
                    if table.get('id'):
                        section_parts.append(f"\n{self.format_table_for_prompt(table)}")
            
            # Add referenced sections with their content and tables
            if ctx['references'] and any(r.get('id') for r in ctx['references']):
                section_parts.append(f"\n--- Referenced Sections ---")
                for ref in ctx['references']:
                    if ref.get('id'):
                        section_parts.append(f"\nReferenced Section: {ref['id']}")
                        if ref.get('title'):
                            section_parts.append(f"Title: {ref['title']}")
                        if ref.get('text'):
                            # Truncate long referenced text
                            ref_text = ref['text']
                            section_parts.append(f"Content: {ref_text}")
                        
                        # Add tables from referenced sections
                        if ref.get('tables') and any(t.get('id') for t in ref['tables']):
                            section_parts.append(f"Tables in {ref['id']}:")
                            for table in ref['tables']:
                                if table.get('id'):
                                    section_parts.append(f"\n{self.format_table_for_prompt(table)}")
            
            section_text = "\n".join(section_parts)
            
            # Check if adding this section would exceed context length
            if current_length + len(section_text) > max_context_length:
                print(f"  Warning: Context length limit reached. Including only {i-1} sections.")
                break
            
            prompt_parts.append(section_text)
            current_length += len(section_text)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("\nBased on the above sections and tables, please provide a comprehensive answer.")
        prompt_parts.append("Always cite specific section numbers and table numbers when referencing requirements.")
        
        full_prompt = "\n".join(prompt_parts)
        prompt_filename = "last_rag_prompt.txt"
        with open(prompt_filename, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        
        # Generate answer using Gemini
        print(f"Generating answer...")
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(full_prompt)
            
            return {
                "answer": response.text,
                "source_sections": [ctx['section_id'] for ctx in contexts],
                "contexts": contexts,
                "prompt_length": len(full_prompt)
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "source_sections": [ctx['section_id'] for ctx in contexts],
                "contexts": contexts,
                "prompt_length": len(full_prompt)
            }


# Usage example
if __name__ == "__main__":
    # Connect and load
    rag = BuildingCodeRAG(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        google_api_key=GOOGLE_API_KEY
    )
    
    try:
        # Ask a question
        question = "I need to better understand how exactly my cantilevered floor joist should be attached to an interior joist. Can you tell me what I need to know?"
        result = rag.answer_question(question, top_k=3)
        
        answer_filename = "last_rag_output.txt"
        with open(answer_filename, 'w', encoding='utf-8') as f:
            f.write(result['answer'])
        
    finally:
        rag.close()