import os
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase

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
            - Associated tables (with full content)
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
                
                // Get tables directly associated with this section
                OPTIONAL MATCH (s)-[:HAS_TABLE]->(t:Table)
                WITH s, parents, references, collect(DISTINCT {
                    id: t.id,
                    name: t.name,
                    content: t.content
                }) as tables
                
                // Get immediate children
                OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(child:Section)
                WITH s, parents, references, tables, collect(DISTINCT {
                    id: child.id,
                    title: child.title
                }) as children
                
                RETURN s.id as section_id,
                       s.title as title,
                       s.text as text,
                       parents,
                       children,
                       references,
                       tables
            """, section_id=section_id)
            
            return result.single()
    
    def format_table_for_prompt(self, table):
        """Format a table for inclusion in the prompt"""
        parts = []
        if table.get('name'):
            parts.append(f"Table: {table['name']}")
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
            
            # Add child sections if any (just titles for context)
            if ctx['children'] and any(c.get('id') for c in ctx['children']):
                child_titles = ", ".join([f"{c['id']} ({c['title']})" for c in ctx['children'] if c.get('id')])
                section_parts.append(f"\nSubsections: {child_titles}")
            
            section_text = "\n".join(section_parts)
            
            # Check if adding this section would exceed context length
            if current_length + len(section_text) > max_context_length:
                print(f"  Warning: Context length limit reached. Including only {i-1} sections.")
                break
            
            prompt_parts.append(section_text)
            current_length += len(section_text)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("\nBased on the above sections and tables, please provide a comprehensive answer.")
        prompt_parts.append("Always cite specific section numbers when referencing requirements.")
        
        full_prompt = "\n".join(prompt_parts)
        
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
    
    def print_context_summary(self, contexts):
        """Helper method to print a summary of retrieved contexts"""
        print("\n" + "="*60)
        print("RETRIEVED CONTEXT SUMMARY")
        print("="*60)
        
        for i, ctx in enumerate(contexts, 1):
            print(f"\nSection {i}: {ctx['section_id']} - {ctx['title']}")
            print(f"  Text length: {len(ctx['text'])} characters")
            print(f"  Tables: {len([t for t in ctx['tables'] if t.get('id')])}")
            print(f"  References: {len([r for r in ctx['references'] if r.get('id')])}")
            
            # Show table names
            if ctx['tables'] and any(t.get('id') for t in ctx['tables']):
                table_names = [t['name'] for t in ctx['tables'] if t.get('name')]
                if table_names:
                    print(f"  Table names: {', '.join(table_names)}")
            
            # Show referenced section IDs
            if ctx['references'] and any(r.get('id') for r in ctx['references']):
                ref_ids = [r['id'] for r in ctx['references'] if r.get('id')]
                print(f"  Referenced sections: {', '.join(ref_ids)}")

# Usage example
if __name__ == "__main__":
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Connect and load
    rag = BuildingCodeRAG(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        google_api_key=GOOGLE_API_KEY
    )
    
    try:
        # Ask a question
        # question = "I have a light-frame construction building and I need to know about the roof members and their limits. I have rooof rafters  that supports gypsum board and I need to know the ratio of the clear span for the rafters"
        # question = "I have a crawl space and I need to understand what the requirements are for drainage. Specifically I need to know whats required to drain the crawl space. The walls are foundation walls."
        question = "I need you to tell me about crawl space drainage and the requirements"
        result = rag.answer_question(question, top_k=3)
        
        # Print context summary
        rag.print_context_summary(result['contexts'])
        
        # Print answer
        print("\n" + "="*60)
        print("ANSWER")
        print("="*60)
        print(result['answer'])
        print(f"\nBased on sections: {', '.join(result['source_sections'])}")
        print(f"Prompt length: {result['prompt_length']} characters")
        
    finally:
        rag.close()