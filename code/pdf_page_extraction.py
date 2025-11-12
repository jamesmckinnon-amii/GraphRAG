import os
import fitz 

INPUT_PATH = "../data/building_code/National_Building_Code2023_Alberta_Edition.pdf"
OUTPUT_PATH = "../data/building_code/Part_9_National_building_Code.pdf"

# Calculate the 0-based indices for the extraction
START_INDEX = 823 - 1
END_INDEX = 1105     


def extract_pdf_page_range_fitz(input_pdf_path: str, output_pdf_path: str, start_index: int, end_index: int):
    """
    Extracts a range of pages from an input PDF using PyMuPDF (fitz) and saves them 
    to a new PDF file.
    """
    # Check if the input file exists
    if not os.path.exists(input_pdf_path):
        print(f"Error: Input file not found at {input_pdf_path}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read and Write the PDF
    try:
        # Open the source PDF document
        source_doc = fitz.open(input_pdf_path)
        total_pages = len(source_doc)
        
        # Check if the requested range is valid
        if start_index < 0 or start_index >= total_pages:
            print(f"Warning: Start page index ({start_index}) is invalid. Setting to 0.")
            start_index = 0
            
        if end_index > total_pages:
            print(f"Warning: End page index ({end_index}) is beyond the document end ({total_pages}). Setting to {total_pages}.")
            end_index = total_pages
        
        if start_index >= end_index:
            print("Error: Invalid or empty page range specified.")
            return

        # Create a new, empty PDF document
        new_doc = fitz.open()

        # Extract pages
        for i in range(start_index, end_index):
            new_doc.insert_pdf(source_doc, from_page=i, to_page=i)
            
        new_doc.save(output_pdf_path)
        new_doc.close()
        source_doc.close()

        print(f"Extracted pages {start_index + 1} to {end_index} (inclusive)")
        print(f"Saved new PDF to:")
        print(f"  {output_pdf_path}")

    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")

if __name__ == "__main__":
    extract_pdf_page_range_fitz(INPUT_PATH, OUTPUT_PATH, START_INDEX, END_INDEX)