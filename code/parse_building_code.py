import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pymupdf4llm
from pathlib import Path

@dataclass
class Section:
    """Represents a section in the building code."""
    number: str
    title: str
    text: str
    start_pos: int
    end_pos: int
    tables: Dict[str, Dict[str, str]] = field(default_factory=dict)
    referenced_text: List[str] = field(default_factory=list)
    
    @property
    def depth(self) -> int:
        """Returns the nesting depth based on number of dots."""
        return self.number.count('.')
    
    @property
    def parent_number(self) -> Optional[str]:
        """Returns the parent section number, or None if top-level."""
        parts = self.number.rstrip('.').split('.')
        if len(parts) <= 2:  # Single dot sections like "9.1." are top-level
            return None
        return '.'.join(parts[:-1]) + '.'


class BuildingCodeParser:
    """Parser for building code markdown documents."""
    
    def __init__(self, markdown_text: str):
        self.text = markdown_text
        self.sections: List[Section] = []
    
    def parse(self) -> Dict:
        """Main parsing method that returns nested structure."""
        self._extract_sections()
        self._extract_content()
        return self._build_hierarchy()
    
    def _extract_sections(self):
        """Extract all section headers from the document (requires trailing dot on section numbers)."""
        # Require trailing dot on the section number to reduce false positives like "9.5 mm ..."
        pattern = re.compile(
            r'^(?:#{2,}\s*(?:Section\s+)?)?(\d+(?:\.\d+)*\.)\s+([^\n]+)',
            re.MULTILINE
        )

        matches = []
        for match in pattern.finditer(self.text):
            number = match.group(1).strip()   # includes trailing dot (e.g. "9.8.1.")
            title = match.group(2).strip()

            # Skip if this looks like a list item (e.g., "1) ...")
            line_start = self.text[max(0, match.start()-5):match.start()]
            if re.search(r'\d+\)\s*$', line_start):
                continue

            matches.append({
                'start': match.start(),
                'end': match.end(),
                'number': number,
                'title': title
            })

        # Filter to only keep sections that are part of the Part 9 hierarchy
        filtered_matches = []
        for m in matches:
            num = m['number'].rstrip('.')  # drop trailing dot for splitting logic
            parts = num.split('.')

            # Must start with 9 and have at least one subsection (e.g., "9.1")
            if parts[0] == '9' and len(parts) >= 2:
                title_lower = m['title'].lower()
                if not any(skip in title_lower for skip in ['notes to table', 'col1', 'col2', 'col3']):
                    filtered_matches.append(m)

        # Sort by position in document
        filtered_matches.sort(key=lambda x: x['start'])

        # Create Section objects
        for match in filtered_matches:
            number = match['number']
            title = match['title']

            # Normalize section number (ensure trailing dot) — regex already requires dot but keep this for safety
            if not number.endswith('.'):
                number += '.'

            section = Section(
                number=number,
                title=title,
                text="",
                start_pos=match['end'],
                end_pos=0
            )
            self.sections.append(section)
    
    def _extract_content(self):
        """Extract text content for each section and capture tables and references."""
        for i, section in enumerate(self.sections):
            # Content ends where the next section begins
            if i + 1 < len(self.sections):
                # Back up to find the actual start of the next section header
                next_section = self.sections[i + 1]
                
                # Search backwards from next section to find start of its header line
                search_start = max(0, next_section.start_pos - 200)
                chunk = self.text[search_start:next_section.start_pos]
                
                # Find the last newline before the section number
                last_newline = chunk.rfind('\n')
                if last_newline != -1:
                    section.end_pos = search_start + last_newline
                else:
                    section.end_pos = next_section.start_pos
            else:
                section.end_pos = len(self.text)
            
            # Extract raw content for the section
            raw_content = self.text[section.start_pos:section.end_pos]
            
            # Extract tables from raw_content and store them in section.tables,
            # removing tables from the text so 'text' doesn't include them.
            raw_without_tables = self._extract_tables_from_section_text(section, raw_content)
            
            # Clean the remaining content
            section.text = self._clean_content(raw_without_tables)
            
            # Extract references from the cleaned text and store them
            raw_refs = self._extract_references(section.text)
            
            # Ensure current section number normalized (trailing dot)
            self_num = section.number if section.number.endswith('.') else section.number + '.'
            
            # Build ancestor set: for section "9.33.8.4." this yields {"9.33.", "9.33.8."}
            parts = self_num.rstrip('.').split('.')
            ancestors = set()
            # include ancestors with at least two dotted groups (skip single-group "9.")
            for i in range(1, len(parts) - 1):
                anc = '.'.join(parts[:i+1]) + '.'
                if anc.count('.') >= 2:  # ensures at least two groups like "9.33."
                    ancestors.add(anc)
            
            # Get the set of table IDs that belong to this section
            own_table_ids = set(section.tables.keys())
            
            # Final filter: keep only refs that are:
            # - not the section itself
            # - not an ancestor section
            # - not a table that belongs to this section
            filtered_refs = []
            for r in raw_refs:
                norm_r = r if r.endswith('.') else (r + '.')
                
                # Skip if it's the section itself
                if norm_r == self_num:
                    continue
                
                # Skip if it's an ancestor
                if norm_r in ancestors:
                    continue
                
                # Skip if it's a table that belongs to this section
                if norm_r in own_table_ids:
                    continue
                
                filtered_refs.append(norm_r)
            
            section.referenced_text = filtered_refs
    
    def _extract_references(self, text: str) -> List[str]:
        """
        Extract references to Articles/Sections/Subsections/Clauses/Tables from a text block.
        Returns a list of normalized dotted references (e.g. "9.10.15.2.", "3.1.5.5.", "9.20.", "Table 9.23.3.4.").
        - Requires at least TWO dotted numeric groups (e.g. "9.24.", "3.1.5.5.") to consider a token valid.
        - Captures labeled references like "Article 9.20.17.5.", "Subsection 9.14.3.", "Clause 3.1.5.5.(1)(b)"
        - Captures table references like "Table 9.23.3.4.", "Tables 9.10.3.1. and 9.10.3.2."
        - Captures additional numbers following a labeled reference separated by commas or 'and'
        - Strips parenthetical sentence/group suffixes and keeps the numeric dotted portion (2 to 4 groups).
        """
        if not text:
            return []
        
        # Pattern for section/article/clause references (requires at least two dotted groups)
        labeled_pattern = re.compile(
            r'\b(?:Article|Articles?|Section|Sections?|Subsection|Subsections?|Clause|Clauses?|'
            r'Subclause|Subclauses?|Sentence|Sentences?)\s+'
            r'((?:\d+\.){2,4}(?:\([^\)]*\))*)',
            re.IGNORECASE
        )
        
        # Pattern specifically for table references (requires at least two dotted groups)
        # Captures both single and multiple table references like "Table 9.10.3.1.-A or 9.10.3.1.-B"
        table_pattern = re.compile(
            r'\bTable(?:s)?\s+((?:\d+\.){2,4}(?:-[A-Z])?(?:\s+(?:or|and)\s+(?:\d+\.){2,4}(?:-[A-Z])?)*)',
            re.IGNORECASE
        )
        
        # Additional dotted numbers (2-4 groups)
        additional_num_pattern = re.compile(r'((?:\d+\.){2,4})')
        
        found_ordered: List[str] = []
        seen = set()
        table_numbers = set()  # Track bare numbers that are from tables
        
        # Extract table references FIRST and mark their numbers
        for m in table_pattern.finditer(text):
            full_match = m.group(1)
            # Extract all table numbers from the match (handles "9.10.3.1.-A or 9.10.3.1.-B")
            table_nums = re.findall(r'((?:\d+\.){2,4})', full_match)
            
            for table_num in table_nums:
                if not table_num.endswith('.'):
                    table_num = table_num + '.'
                
                table_ref = f"Table {table_num}"
                table_numbers.add(table_num)  # Mark this bare number as belonging to a table
                
                if table_ref not in seen:
                    seen.add(table_ref)
                    found_ordered.append(table_ref)
        
        # Extract section/article/clause references
        # Skip any numbers that were already identified as table references
        for m in labeled_pattern.finditer(text):
            group = m.group(1)
            # Extract leading dotted numeric groups (2-4 groups)
            lead = re.match(r'((?:\d+\.){2,4})', group)
            if lead:
                normalized = lead.group(1)
                if not normalized.endswith('.'):
                    normalized = normalized + '.'
                
                # Skip if this number belongs to a table
                if normalized in table_numbers:
                    continue
                
                if normalized not in seen:
                    seen.add(normalized)
                    found_ordered.append(normalized)
            
            # Lookahead slice for additional numbers that might follow without repeated label
            lookahead_span_end = min(len(text), m.end() + 200)
            tail = text[m.end():lookahead_span_end]
            # Find numbers directly following like ", 6.2.1.1." or "and 6.2.1.1."
            for add in additional_num_pattern.findall(tail):
                add_norm = add
                if not add_norm.endswith('.'):
                    add_norm = add_norm + '.'
                
                # Skip if this number belongs to a table
                if add_norm in table_numbers:
                    continue
                
                if add_norm not in seen:
                    seen.add(add_norm)
                    found_ordered.append(add_norm)
        
        # Capture bare dotted references near 'see' or parentheses but requiring 2+ groups
        # This handles cases like "(see 9.20., 9.27. or 9.28.)"
        loose_pattern = re.compile(
            r'(?:(?:see|see also|see Article|see Section)\b.{0,80}?|\()\s*(((?:\d+\.){2,4}))',
            re.IGNORECASE
        )
        for lm in loose_pattern.finditer(text):
            tkn = lm.group(1)
            if tkn:
                if not tkn.endswith('.'):
                    tkn = tkn + '.'
                
                # Skip if this number belongs to a table
                if tkn in table_numbers:
                    continue
                
                if tkn not in seen:
                    seen.add(tkn)
                    found_ordered.append(tkn)
        
        return found_ordered


    def _extract_tables_from_section_text(self, section: Section, raw: str) -> str:
        """
        Finds markdown tables inside the raw section text, associates a caption
        if possible (including bold/plain 'Table <num>' even if separated by blank lines),
        stores them in section.tables (with 'table_name' and 'table_content'),
        removes the table block and nearby caption/notes lines from the text,
        and returns the raw text with those removed.
        """
        def clean_title_text(s: str) -> str:
            s = re.sub(r'<br\s*/?>', ' ', s, flags=re.IGNORECASE)
            s = re.sub(r'[_\*]{1,2}', '', s)          # remove markdown emphasis/bold markers
            s = re.sub(r'^[\s\.\-\:]+', '', s)        # strip leading punctuation like ".", "-", ":"
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        # Regex for a markdown table block
        table_regex = re.compile(
            r'(^\|[^\n]*\|\s*\n'                # header row
            r'^\|(?:\s*[:-]+[-\s|:]*)\|\s*\n'  # divider row
            r'(?:^\|[^\n]*\|\s*\n?)+)',        # one or more data rows
            re.MULTILINE
        )

        cleaned = raw
        matches = list(table_regex.finditer(raw))
        if not matches:
            return raw

        # caption-like line detection regex (table captions, "Notes to Table", bolded forms)
        caption_line_re = re.compile(
            r'^\s*(?:\*{0,2}\s*Notes to Table\b|Table\s+[0-9A-Za-z\.\-]+\.?|Table\b).*$', re.IGNORECASE
        )

        for match in reversed(matches):
            table_text = match.group(0).rstrip('\n')
            start_idx = match.start()
            end_idx = match.end()

            # Inspect header row for "Table <num> Title" token ---
            header_line = ""
            for ln in table_text.splitlines():
                if ln.strip():
                    header_line = ln.strip()
                    break
            # split header_line into cells (keep empties)
            cells = [c for c in header_line.split('|')]

            table_number_key = None
            table_name = ""
            found_in_header = False

            # look for 'Table <num>' in any single header cell (preferred)
            for cell in cells:
                cell_text = cell.strip()
                if not cell_text:
                    continue
                m = re.search(r'(?i)\bTable\s+([0-9A-Za-z\.\-]+\.?)\b', cell_text)
                if m:
                    raw_num = m.group(1).strip()
                    after = cell_text[m.end():].strip()
                    if not raw_num.endswith('.'):
                        num_norm = raw_num + '.'
                    else:
                        num_norm = raw_num
                    table_number_key = f"Table {num_norm}"
                    table_name = clean_title_text(after)
                    found_in_header = True
                    break

            # If not found in header, search preceding context for caption ---
            caption_region_start = None
            if not found_in_header:
                lookback = 1200  # chars before table to search for caption
                lookback_slice_start = max(0, start_idx - lookback)
                context_before = raw[lookback_slice_start:start_idx]

                # Split preceding context into lines and reverse to search backward
                lines = context_before.splitlines()
                lines.reverse()

                # Find the first line with "Table <num>"
                table_line_idx = None
                for idx, line in enumerate(lines):
                    m_table = re.search(r'(?i)Table\s+([0-9A-Za-z\.\-]+\.?)', line)
                    if m_table:
                        raw_num = m_table.group(1).strip()
                        if not raw_num.endswith('.'):
                            num_norm = raw_num + '.'
                        else:
                            num_norm = raw_num
                        table_number_key = f"Table {num_norm}"
                        table_line_idx = idx
                        break

                # Collect lines above the table line as the table title 
                if table_line_idx is not None:
                    table_title_lines = []
                    # lines[:table_line_idx] are the lines above the 'Table...' line
                    for title_line in lines[:table_line_idx]:
                        title_line = title_line.strip()
                        if not title_line:
                            continue
                        # stop if we see another Table line
                        if re.search(r'(?i)Table\s+[0-9A-Za-z\.\-]+\.?', title_line):
                            break
                        table_title_lines.append(title_line)
                    table_title_lines.reverse()  # restore correct order
                    table_name = clean_title_text(' '.join(table_title_lines))

                    # compute absolute start index of the caption line block to remove it along with the table
                    # find line positions in the original raw
                    # find the position of the 'Table' line (first occurrence going backwards)
                    # We'll search from start_idx backwards to find the 'Table' line start
                    search_pos = start_idx
                    # Scan back up to lookback chars for the Table line
                    preceding_slice = raw[lookback_slice_start:start_idx]
                    # find the last occurrence of the "Table <num>" inside preceding_slice
                    m_last_table = None
                    for mtmp in re.finditer(r'(?i)Table\s+[0-9A-Za-z\.\-]+\.?', preceding_slice):
                        m_last_table = mtmp
                    if m_last_table:
                        # absolute position
                        abs_table_pos = lookback_slice_start + m_last_table.start()
                        caption_region_start = abs_table_pos
                    else:
                        caption_region_start = None

            # Fallbacks: if neither header nor caption gave a number, make alternate key ---
            if not table_number_key:
                # Try to find "Table <num>" anywhere in header_line
                m2 = re.search(r'(?i)Table\s+([0-9A-Za-z\.\-]+\.?)', header_line)
                if m2:
                    raw_num = m2.group(1).strip()
                    if not raw_num.endswith('.'):
                        num_norm = raw_num + '.'
                    else:
                        num_norm = raw_num
                    table_number_key = f"Table {num_norm}"
                    after = header_line[m2.end():].strip()
                    table_name = clean_title_text(after)
                else:
                    # final synthetic key built from a short header snippet
                    safe_key = (header_line[:60].strip() or "unnamed").strip()
                    table_number_key = f"Table: {safe_key}"
                    table_name = ""

            # Ensure unique keys
            base_key = table_number_key or "Table"
            unique_key = base_key
            counter = 1
            while unique_key in section.tables:
                counter += 1
                unique_key = f"{base_key} ({counter})"

            # Store table dict
            section.tables[unique_key] = {
                'table_name': table_name,
                'table_content': table_text
            }

            # Remove the table block from the text (replace with a single newline)
            # Additionally remove caption/notes lines immediately above the table (if any) to avoid capturing their numbers.
            remove_start = match.start()
            # Only attempt to remove caption lines if we identified a caption_region_start earlier,
            # otherwise look a few lines above and remove contiguous caption-like lines.
            if caption_region_start is not None:
                remove_start = caption_region_start
            else:
                # scan up to 10 lines above the table start to find contiguous caption-like lines
                # compute the slice of raw that precedes the table
                scan_limit = 10
                pos = start_idx
                lines_removed = 0
                while lines_removed < scan_limit:
                    prev_nl = raw.rfind('\n', 0, pos)
                    if prev_nl == -1:
                        line_start = 0
                    else:
                        line_start = prev_nl + 1
                    line = raw[line_start:pos].strip()
                    if not line:
                        # blank line - include it but keep scanning further up
                        remove_start = line_start
                        pos = line_start
                        lines_removed += 1
                        continue
                    # if the line looks like a caption or "Notes to Table", mark it for removal
                    if caption_line_re.match(line):
                        remove_start = line_start
                        pos = line_start
                        lines_removed += 1
                        continue
                    # if the line is short and starts with bold or table tokens, remove it too
                    if len(line) <= 80 and ('Table' in line or 'Notes to Table' in line or line.startswith('**')):
                        remove_start = line_start
                        pos = line_start
                        lines_removed += 1
                        continue
                    # otherwise stop scanning
                    break

            # perform removal on the cleaned text
            cleaned = cleaned[:remove_start] + "\n" + cleaned[end_idx:]

        return cleaned



    
    def _clean_content(self, content: str) -> str:
        """Clean up extracted content by removing headers and extra whitespace."""
        # Remove lines that start with section numbers (subsections that were captured)
        content = re.sub(r'^\d+(?:\.\d+)*\.?\s+[^\n]+$', '', content, flags=re.MULTILINE)
        
        # Remove lines that are just markdown formatting
        content = re.sub(r'^\s*[#\*_\-]+\s*$', '', content, flags=re.MULTILINE)
        
        # Remove page numbers and footers (e.g., "_**9-1**_")
        content = re.sub(r'^\s*_?\*{0,2}\d+-\d+\*{0,2}_?\s*$', '', content, flags=re.MULTILINE)
        
        # Remove section headers that might have been included
        content = re.sub(r'^#{2,}\s+.*$', '', content, flags=re.MULTILINE)
        
        # Remove any leftover table lines just in case (defensive)
        content = re.sub(r'^\|.*\|$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*[-\s\|:]{3,}\s*$', '', content, flags=re.MULTILINE)
        
        # Consolidate multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _build_hierarchy(self) -> Dict:
        """Build nested dictionary structure from flat section list."""
        root = {}
        
        for section in self.sections:
            self._insert_section(root, section)
        
        return root
    
    def _insert_section(self, root: Dict, section: Section):
        """Insert a section into the hierarchy at the correct location."""
        parts = section.number.rstrip('.').split('.')
        
        # If this is a top-level section (like 9.1), add directly to root
        if len(parts) == 2:  # e.g., "9.1"
            root[section.number] = {
                'title': section.title,
                'text': section.text,
                'tables': section.tables,
                'referenced_text': section.referenced_text,
                'subsections': {}
            }
            return
        
        # For nested sections, navigate to parent and add there
        current = root
        for i in range(1, len(parts) - 1):  # Start from 1 to skip "9"
            parent_key = '.'.join(parts[:i+1]) + '.'
            
            # If parent doesn't exist, we can't place this section
            if parent_key not in current:
                print(f"Warning: Parent '{parent_key}' not found for section '{section.number}'")
                return
            
            current = current[parent_key]['subsections']
        
        # Insert the section at the correct location
        current[section.number] = {
            'title': section.title,
            'text': section.text,
            'tables': section.tables,
            'referenced_text': section.referenced_text,
            'subsections': {}
        }


def convert_pdf_to_hierarchy(pdf_path: str) -> Dict:
    """
    Convert a building code PDF to a nested hierarchy dictionary.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with nested section structure
    """
    print(f"Converting PDF: {pdf_path} to Markdown...")
    # convert to markdown if markdown does not already exist
    markdown_path = Path("../data/processed/Part_9_National_building_Code.md")
    if not markdown_path.exists():
        markdown_content = pymupdf4llm.to_markdown(doc=pdf_path)
        markdown_path.write_text(markdown_content, encoding="utf-8")
    else:
        markdown_content = markdown_path.read_text(encoding="utf-8")
    
    print("Parsing building code structure...")
    parser = BuildingCodeParser(markdown_content)
    hierarchy = parser.parse()
    
    print(f"\nTotal sections parsed: {len(parser.sections)}")
    
    # Count sections at each level
    level_counts = {}
    for section in parser.sections:
        depth = section.depth
        level_counts[depth] = level_counts.get(depth, 0) + 1
    
    print("Section breakdown by depth:")
    for depth in sorted(level_counts.keys()):
        print(f"  Level {depth}: {level_counts[depth]} sections")
    
    return hierarchy


if __name__ == "__main__":
    # Configuration
    input_pdf_path = "../data/building_code/Part_9_National_building_Code.pdf"
    output_json_path = "../data/processed/Part_9_National_building_Code.json"
    
    # Parse the document
    code_hierarchy = convert_pdf_to_hierarchy(input_pdf_path)
    
    # Save to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(code_hierarchy, f, indent=4, ensure_ascii=False)
    
    print(f"\nSaved hierarchy to {output_json_path}")