import os
import pydicom
import SimpleITK as sitk
import pandas as pd
import numpy as np
import zipfile
import glob
import shutil
import tempfile
import argparse
import re
from PIL import Image, ImageDraw
try:
    import fitz  # PyMuPDF
    import pytesseract
    ADVANCED_PDF_AVAILABLE = True
except ImportError:
    ADVANCED_PDF_AVAILABLE = False
    print("Note: Advanced PDF processing (PyMuPDF/pytesseract) not available. Using basic text extraction.")
import PyPDF2



def unzip_file(zip_path, extract_path=None):
    """
    Extract a zip file to a specified directory.
    Returns paths to ALL folders containing DICOM slices and annotation file.
    """
    # If no extraction path is specified, use the current directory
    if extract_path is None:
        extract_path = os.path.dirname(zip_path)
        extract_name = os.path.basename(zip_path).split(".")[0]
        extract_path = os.path.join(extract_path, extract_name)

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # List to store extracted file names
    extracted_files = []
    dicom_folder_paths = []  # Changed to list to store multiple paths
    annotation_path = None

    try:
        # Open the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all contents
            zip_ref.extractall(extract_path)
            # Get list of all files in the ZIP
            extracted_files = zip_ref.namelist()

        # Look for DICOM folders and annotation file
        for filename in glob.iglob(extract_path + '**/**', recursive=True):
            if os.path.isfile(filename):
                # Check for annotation files (prioritize PDF for reports)
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in [".txt", ".pdf", ".jpeg", ".jpg", ".png"] and annotation_path is None:
                    annotation_path = filename
                elif file_ext == ".pdf" and annotation_path is not None:
                    # If we find a PDF and already have another annotation, prefer the PDF
                    annotation_path = filename
            elif os.path.isdir(filename):  # Removed "and dicom_folder_path is None"
                # Check if this directory contains DICOM files
                try:
                    dir_files = os.listdir(filename)
                    for file in dir_files:
                        file_path = os.path.join(filename, file)
                        if os.path.isfile(file_path):
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext in ['.dcm', '.dicom', '.ima'] or (file_ext == '' and is_dicom_file(file_path)):
                                if filename not in dicom_folder_paths:  # Avoid duplicates
                                    dicom_folder_paths.append(filename)
                                    print(f"Found DICOM directory: {filename}")
                                break
                except Exception as e:
                    continue

        return dicom_folder_paths, annotation_path  # Return list of paths

    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file")
        return [], None  # Return empty list instead of None
    except Exception as e:
        print(f"Error occurred while extracting: {str(e)}")
        return [], None  # Return empty list instead of None


def load_dicom_file(dicom_path):
    """
    Load a single DICOM file using SimpleITK.
    """
    # Ensure the path points to a single file
    if not os.path.isfile(dicom_path):
        raise ValueError(f"The path '{dicom_path}' does not point to a valid file.")

    # Load the DICOM file
    image = sitk.ReadImage(dicom_path)
    image_size = image.GetSize()
    print(f"Image shape (loaded): {image_size}")
    return image


# Remove PHI from DICOM metadata
tags_to_remove = [# Patient Identifiers
                  "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
                  "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
                  "PatientBirthTime", "PatientSex", "PatientAge", "PatientSize",
                  "PatientWeight",

                  # Study and Clinical Information
                  "AccessionNumber", "StudyID", "StudyDate", "StudyTime",
                  "ReferringPhysicianName", "InstitutionName", "RequestingPhysician",
                  "StudyDescription", "SeriesDescription",

                  # Additional Identifying Elements
                  "OperatorsName", "PhysicianOfRecord", "PerformingPhysicianName",
                  "PersonName", "IssuerOfPatientID", "NameOfPhysiciansReadingStudy",

                  # Date and Time Information
                  "ContentDate", "ContentTime", "AcquisitionDate", "AcquisitionTime",
                  "InstanceCreationDate", "InstanceCreationTime",

                  # Private Tags and Comments
                  "ImageComments", "AcquisitionComments",

                  # Additional possible identifiers
                  "PatientInsurancePlanCodeSequence", "MilitaryRank", "BranchOfService",
                  "MedicalRecordLocator", "MedicalAlerts", "Allergies",
                  "CountryOfResidence", "RegionOfResidence", "EthnicGroup",
                  "Occupation", "AdditionalPatientHistory", "PregnancyStatus"]


def is_dicom_file(file_path):
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False


def discover_dicom_structure(root_directory):
    """
    Discover the directory structure and find all DICOM files organized by subdirectories.
    Returns a dictionary with subdirectory names as keys and lists of DICOM files as values.
    """
    dicom_structure = {}
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(root_directory):
        # Get relative path from root directory
        relative_path = os.path.relpath(root, root_directory)
        
        # Skip the root directory itself
        if relative_path == '.':
            continue
            
        # Find DICOM files in this directory
        dicom_files = []
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check common DICOM extensions and files without extension
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ['.dcm', '.dicom', '.ima'] or file_ext == '':
                if is_dicom_file(file_path):
                    dicom_files.append(file_path)
        
        # If we found DICOM files, add this directory to our structure
        if dicom_files:
            dicom_structure[relative_path] = sorted(dicom_files)
            print(f"Found {len(dicom_files)} DICOM files in subdirectory: {relative_path}")
    
    return dicom_structure


def get_dicom_files(directory):
    """Get all DICOM files from a directory (original function for backward compatibility)."""
    dicom_files = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        if os.path.isdir(file_path):
            continue

        # Check common DICOM extensions and files without extension
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in ['.dcm', '.dicom', '.ima'] or file_ext == '':
            if is_dicom_file(file_path):
                dicom_files.append(file_path)

    return sorted(dicom_files)


def deidentify_pixel_data(ds):
    """
    Black out the top-left corner of DICOM image to remove burned-in PHI.
    Covers 1/6 of width and 1/15 of height.
    
    Parameters:
    -----------
    ds : pydicom.Dataset
        DICOM dataset to modify
    """
    try:
        if hasattr(ds, 'PixelData'):
            # Get image dimensions
            rows = ds.Rows
            cols = ds.Columns
            
            # Calculate dimensions to black out
            black_width = cols // 6
            black_height = rows // 15
            
            # Try to get pixel array
            try:
                pixel_array = ds.pixel_array
                
                # Black out the top-left corner
                pixel_array[:black_height, :black_width] = 0
                
                # Handle compressed and uncompressed data differently
                if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
                    # Check if transfer syntax is compressed
                    compressed_syntaxes = [
                        '1.2.840.10008.1.2.4.50',  # JPEG Baseline
                        '1.2.840.10008.1.2.4.51',  # JPEG Extended
                        '1.2.840.10008.1.2.4.57',  # JPEG Lossless
                        '1.2.840.10008.1.2.4.70',  # JPEG Lossless First Order
                        '1.2.840.10008.1.2.4.80',  # JPEG-LS Lossless
                        '1.2.840.10008.1.2.4.81',  # JPEG-LS Lossy
                        '1.2.840.10008.1.2.4.90',  # JPEG 2000 Lossless
                        '1.2.840.10008.1.2.4.91',  # JPEG 2000
                        '1.2.840.10008.1.2.5',     # RLE Lossless
                    ]
                    
                    if str(ds.file_meta.TransferSyntaxUID) in compressed_syntaxes:
                        # For compressed data, we need to decompress, modify, and recompress
                        # For now, skip pixel modification for compressed data
                        return False
                
                # For uncompressed data, update pixel data
                ds.PixelData = pixel_array.tobytes()
                return True
                
            except Exception as e:
                # If we can't get pixel array (e.g., compressed data), skip modification
                return False
                
    except Exception as e:
        # If any error occurs, skip pixel modification
        return False
    
    return False


def remove_phi_metadata_structured(input_path, output_dir):
    """
    Remove PHI metadata tags from DICOM files while preserving subdirectory structure.
    
    Parameters:
    -----------
    input_path : str
        Path to directory containing DICOM files (potentially in subdirectories)
    output_dir : str
        Directory where de-identified DICOM files will be saved with preserved structure
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing DICOM data from: {input_path}")
    print(f"Output will be saved to: {output_dir}")

    # Discover the DICOM structure
    if os.path.isdir(input_path):
        dicom_structure = discover_dicom_structure(input_path)
        
        if not dicom_structure:
            # Fallback to old method for flat structure
            print("No subdirectories with DICOM files found, checking root directory...")
            dicom_files = get_dicom_files(input_path)
            if dicom_files:
                dicom_structure = {'root': dicom_files}
            else:
                print("No DICOM files found in any directory")
                return []
    else:
        if is_dicom_file(input_path):
            dicom_structure = {'single_file': [input_path]}
        else:
            print(f"Error: {input_path} is not a valid DICOM file")
            return []

    total_files = sum(len(files) for files in dicom_structure.values())
    print(f"Found {total_files} valid DICOM files across {len(dicom_structure)} subdirectories")

    processed_file_paths = []
    overall_count = 0

    # Process each subdirectory
    for subdir, dicom_files in dicom_structure.items():
        print(f"\nProcessing subdirectory: {subdir}")
        
        # Create output subdirectory
        if subdir == 'root':
            subdir_output = output_dir
        elif subdir == 'single_file':
            subdir_output = output_dir
        else:
            subdir_output = os.path.join(output_dir, subdir)
            os.makedirs(subdir_output, exist_ok=True)
        
        processed_count = 0
        
        for i, file_path in enumerate(dicom_files):
            try:
                # Load DICOM file
                ds = pydicom.dcmread(file_path)

                # Remove PHI metadata tags
                removed_tags = []
                for tag in tags_to_remove:
                    if hasattr(ds, tag):
                        delattr(ds, tag)
                        removed_tags.append(tag)
                
                # De-identify pixel data (black out top-left corner)
                pixel_deidentified = deidentify_pixel_data(ds)

                # Use generic filename to avoid PHI in filenames
                output_filename = f"image_{i+1:04d}.dcm"
                output_file = os.path.join(subdir_output, output_filename)

                # Save the de-identified file
                ds.save_as(output_file)
                processed_count += 1
                overall_count += 1
                processed_file_paths.append(output_file)

                print(f"  ✓ Slice {i+1} -> {output_filename} (removed {len(removed_tags)} PHI tags, pixel data: {'modified' if pixel_deidentified else 'unchanged'})")

            except Exception as e:
                print(f"  ✗ Failed to process slice {i+1}: {str(e)}")

        print(f"  Subdirectory summary: {processed_count}/{len(dicom_files)} files processed")

    print(f"\nOverall summary: Successfully processed {overall_count} out of {total_files} files")
    print(f"De-identified files saved to {output_dir} with preserved directory structure")

    return processed_file_paths


def remove_phi_metadata(input_path, output_dir):
    """
    Remove PHI metadata tags from DICOM files.
    This is the updated version that handles both flat and nested structures.
    """
    return remove_phi_metadata_structured(input_path, output_dir)


def deidentify_annotation(annotation_path):
    """
    De-identify an annotation file by removing PHI from text content.
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        deidentified_lines = []
        for line in lines:
            # Simple de-identification: remove patterns that could be PHI
            import re

            # Remove names (assuming they are capitalized words)
            line = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME REDACTED]', line)

            # Remove dates
            line = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE REDACTED]', line)
            line = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DATE REDACTED]', line)

            # Remove phone numbers
            line = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE REDACTED]', line)

            # Remove potential ID numbers
            line = re.sub(r'\b\d{6,}\b', '[ID REDACTED]', line)

            deidentified_lines.append(line)

        return deidentified_lines

    except Exception as e:
        print(f"Error de-identifying annotation file: {str(e)}")
        return []


def get_annotation_file_type(file_path):
    """Get the file extension to determine the type of annotation file."""
    return os.path.splitext(file_path)[1].lower()


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file and format it with logical line breaks."""
    try:
        import PyPDF2
        import re
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        # Add logical line breaks for better formatting
        # Split on common field patterns and section headers
        text = re.sub(r'(\s+)(PATIENT\'S NAME|DATE|AGE / SEX|CT ID|REF BY|INVESTIGATION)', r'\n\1\2', text)
        text = re.sub(r'(\s+)(CT REPORT|TECHNIQUE|OBSERVATION|IMPRESSION)', r'\n\n\1\2', text)
        text = re.sub(r'(\s+)(●)', r'\n\1\2', text)  # Bullet points
        text = re.sub(r'(\.)(\s+)(Note made of|There is)', r'\1\n\2\3', text)  # Sentence breaks
        
        # Clean up excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading spaces from lines
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""


def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        return text
    except Exception as e:
        print(f"Error extracting text from image: {str(e)}")
        return ""


def deidentify_text_by_rows(text):
    """
    De-identify text by processing each field/row separately.
    Remove entire rows containing PHI but preserve clinical impressions.
    """
    if not text:
        return text
    
    import re
    
    # Split text into lines/rows
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        original_line = line  # Keep original line with whitespace
        line_stripped = line.strip()
        if not line_stripped:  # Keep empty lines for formatting
            cleaned_lines.append(original_line)
            continue
        
        # Check if this row contains clinical information we want to preserve
        clinical_keywords = [
            'impression', 'findings', 'conclusion', 'diagnosis', 'assessment',
            'recommendation', 'opinion', 'interpretation', 'observation',
            'examination', 'scan', 'study', 'image', 'radiolog', 'clinical',
            'pathology', 'abnormal', 'normal', 'lesion', 'mass', 'tumor',
            'enhancement', 'contrast', 'anatomy', 'organ', 'tissue',
            'procedure', 'technique', 'protocol', 'modality'
        ]
        
        line_lower = line_stripped.lower()
        is_clinical = any(keyword in line_lower for keyword in clinical_keywords)
        
        # If it's clinical content, keep it but still clean any embedded PHI
        if is_clinical:
            # Clean embedded PHI patterns but keep the line
            cleaned_line = original_line
            # Remove specific PHI patterns while preserving clinical context
            cleaned_line = re.sub(r'\b\d{1,3}\s*(?:years?|yrs?|y\.?o\.?)\s*(?:old)?\b', '[AGE]', cleaned_line, flags=re.IGNORECASE)
            cleaned_line = re.sub(r'\b(?:male|female|m|f)\b(?!\s*(?:doctor|physician|nurse|patient))', '[SEX]', cleaned_line, flags=re.IGNORECASE)
            # Remove dates but be careful not to remove medical dates that are important
            cleaned_line = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', cleaned_line)
            cleaned_lines.append(cleaned_line)
            continue
        
        # Check if this row contains PHI that should be removed entirely
        phi_patterns = [
            # Names (but be careful not to remove medical terms)
            r'\b(patient\s+name|name)\s*:',
            r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b',  # Potential names
            
            # Age/Sex information
            r'\b(age|sex|gender)\s*:',
            r'\b\d{1,3}\s*(?:years?|yrs?|y\.?o\.?)\s*(?:old)?\b',
            r'\b(?:male|female|m|f)\s*(?:patient)?\b',
            
            # Address information
            r'\b(address|street|city|state|zip|postal)\s*:',
            r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|Boulevard)\b',
            
            # ID numbers
            r'\b(patient\s+id|id|medical\s+record|mrn|ssn)\s*:',
            r'\b\d{6,}\b',  # Long number sequences
            
            # Phone numbers
            r'\b(phone|telephone|contact)\s*:',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            
            # Dates of birth
            r'\b(date\s+of\s+birth|dob|birth\s+date)\s*:',
            
            # Doctor/referring physician names
            r'\b(referring\s+(?:doctor|physician)|doctor|physician)\s*:',
        ]
        
        # Check if this line contains PHI patterns
        contains_phi = False
        for pattern in phi_patterns:
            if re.search(pattern, line_lower):
                contains_phi = True
                break
        
        # If line contains PHI, skip it entirely
        if contains_phi:
            continue
        else:
            # Keep the line as it doesn't contain obvious PHI (preserve original formatting)
            cleaned_lines.append(original_line)
    
    return '\n'.join(cleaned_lines)


def deidentify_text(text):
    """
    De-identify text by removing PHI while preserving medical content.
    Conservative approach - only removes content after specific PHI labels.
    """
    if not text:
        return text
    
    lines = text.split('\n')
    deidentified_lines = []
    
    for line in lines:
        # Remove content after specific PHI labels
        import re
        
        # Pattern for labels followed by sensitive information
        phi_patterns = [
            r'(Patient Name|Name)\s*:.*',
            r'(Referring Doctor|Doctor|Physician)\s*:.*',
            r'(Date of Birth|DOB|Birth Date)\s*:.*',
            r'(Patient ID|ID|Medical Record)\s*:.*',
            r'(Phone|Telephone|Contact)\s*:.*',
            r'(Address)\s*:.*',
            r'(Age|Sex)\s*:\s*\S+',  # Only redact the value, not the label
        ]
        
        cleaned_line = line
        for pattern in phi_patterns:
            cleaned_line = re.sub(pattern, lambda m: m.group(0).split(':')[0] + ': [REDACTED]', cleaned_line, flags=re.IGNORECASE)
        
        # Clean up specific age/sex patterns but preserve medical context
        cleaned_line = re.sub(r'\b\d{1,3}\s*(?:years?|yrs?|y\.?o\.?)\s*(?:old)?\b', '[AGE]', cleaned_line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r'\b(?:male|female|m|f)\b(?!\s*(?:doctor|physician|nurse))', '[SEX]', cleaned_line, flags=re.IGNORECASE)
        
        deidentified_lines.append(cleaned_line)
    
    return '\n'.join(deidentified_lines)


def identify_phi_patterns(text):
    """
    Identify PHI patterns in text and return their positions and types.
    Returns list of tuples: (start_pos, end_pos, phi_type)
    """
    phi_matches = []
    
    # Define PHI patterns with their types
    phi_patterns = [
        # Names (capitalized words that could be names)
        (r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', 'NAME'),
        
        # Dates (various formats)
        (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DATE'),
        (r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', 'DATE'),
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', 'DATE'),
        
        # Phone numbers
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 'PHONE'),
        (r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b', 'PHONE'),
        
        # Medical Record Numbers / IDs
        (r'\b(?:MRN|ID|SSN)\s*:?\s*\d+\b', 'ID'),
        (r'\b\d{6,}\b', 'ID'),  # Long number sequences
        
        # Ages with years
        (r'\b\d{1,3}\s*(?:years?|yrs?|y\.?o\.?)\s*(?:old)?\b', 'AGE'),
        
        # Addresses (basic patterns)
        (r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|Boulevard)\b', 'ADDRESS'),
        
        # Email addresses
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
    ]
    
    for pattern, phi_type in phi_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Skip common medical terms that might match name patterns
            matched_text = match.group().lower()
            medical_terms = ['patient', 'doctor', 'physician', 'nurse', 'clinic', 'hospital', 'medical', 'radiology', 'imaging']
            
            if phi_type == 'NAME' and any(term in matched_text for term in medical_terms):
                continue
                
            phi_matches.append((match.start(), match.end(), phi_type))
    
    return phi_matches


def redact_pdf_with_ocr(pdf_path, output_path):
    """
    Process a PDF file by using OCR to identify sensitive information 
    and placing black boxes over the text.
    """
    if ADVANCED_PDF_AVAILABLE:
        return _redact_pdf_with_pymupdf(pdf_path, output_path)
    else:
        return _redact_pdf_basic(pdf_path, output_path)


def _redact_pdf_with_pymupdf(pdf_path, output_path):
    """
    Advanced PDF redaction using PyMuPDF and pytesseract.
    """
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        print(f"Processing PDF: {os.path.basename(pdf_path)} ({len(pdf_document)} pages)")
        
        redaction_count = 0
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convert page to image for OCR
            mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Load image with PIL
            image = Image.open(Image.io.BytesIO(img_data))
            
            # Perform OCR to get text with bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text and analyze for PHI
            full_text = " ".join([word for word in ocr_data['text'] if word.strip()])
            phi_matches = identify_phi_patterns(full_text)
            
            if phi_matches:
                print(f"  Page {page_num + 1}: Found {len(phi_matches)} potential PHI instances")
                
                # Find words that match PHI patterns
                current_pos = 0
                word_positions = []
                
                for i, word in enumerate(ocr_data['text']):
                    if word.strip():
                        word_start = current_pos
                        word_end = current_pos + len(word)
                        word_positions.append((word_start, word_end, i))
                        current_pos = word_end + 1  # +1 for space
                
                # Create redaction rectangles
                for phi_start, phi_end, phi_type in phi_matches:
                    # Find OCR words that overlap with PHI text
                    for word_start, word_end, word_idx in word_positions:
                        if (word_start < phi_end and word_end > phi_start):
                            # Get word bounding box from OCR
                            left = ocr_data['left'][word_idx]
                            top = ocr_data['top'][word_idx]
                            width = ocr_data['width'][word_idx]
                            height = ocr_data['height'][word_idx]
                            
                            # Convert coordinates back to PDF coordinate system
                            # OCR was done on 2x scaled image, so divide by 2
                            pdf_left = left / 2.0
                            pdf_top = top / 2.0
                            pdf_width = width / 2.0
                            pdf_height = height / 2.0
                            
                            # Create redaction rectangle (PyMuPDF uses bottom-left origin)
                            rect = fitz.Rect(
                                pdf_left,
                                pdf_top,
                                pdf_left + pdf_width,
                                pdf_top + pdf_height
                            )
                            
                            # Add redaction annotation
                            page.add_redact_annot(rect)
                            redaction_count += 1
                
                # Apply redactions (this blacks out the areas)
                page.apply_redactions()
        
        # Save the redacted PDF
        pdf_document.save(output_path)
        pdf_document.close()
        
        print(f"  Applied {redaction_count} redactions and saved to: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return False


def _redact_pdf_basic(pdf_path, output_path):
    """
    Basic PDF processing using PyPDF2. Creates a text-based redacted version.
    """
    try:
        print(f"Processing PDF (basic mode): {os.path.basename(pdf_path)}")
        
        # Extract text from PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract all text
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        
        # Identify PHI patterns
        phi_matches = identify_phi_patterns(full_text)
        
        if phi_matches:
            print(f"  Found {len(phi_matches)} potential PHI instances")
            
            # Create redacted text by replacing PHI with [REDACTED]
            redacted_text = full_text
            
            # Sort matches by position (reverse order to avoid position shifting)
            phi_matches.sort(key=lambda x: x[0], reverse=True)
            
            for start_pos, end_pos, phi_type in phi_matches:
                redacted_text = redacted_text[:start_pos] + f"[{phi_type} REDACTED]" + redacted_text[end_pos:]
        else:
            print("  No PHI patterns detected")
            redacted_text = full_text
        
        # Save redacted text as a new text file (since we can't easily modify PDF visually)
        text_output_path = output_path.replace('.pdf', '_redacted.txt')
        with open(text_output_path, 'w', encoding='utf-8') as file:
            file.write("=== REDACTED PDF CONTENT ===\n")
            file.write("(Original PDF processed with basic text extraction)\n\n")
            file.write(redacted_text)
        
        # Also copy the original PDF for reference
        import shutil
        shutil.copy2(pdf_path, output_path)
        
        print(f"  Basic redaction complete:")
        print(f"    - Original PDF copied to: {os.path.basename(output_path)}")
        print(f"    - Redacted text saved to: {os.path.basename(text_output_path)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return False


def process_annotation_file(annotation_path, output_dir):
    """
    Process annotation file: extract text if needed, deidentify, and save as .txt file.
    For PDFs, also create a redacted PDF with black boxes over sensitive information.
    """
    file_type = get_annotation_file_type(annotation_path)
    
    # Extract text based on file type
    if file_type == '.txt':
        # Already text file, just read it
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        # Use row-based deidentification that preserves clinical impressions
        deidentified_text = deidentify_text_by_rows(text)
        
        # Save as .txt file
        output_path = os.path.join(output_dir, "annotation_deidentified.txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(deidentified_text)
        
        print(f"Processed annotation file saved to: {output_path}")
        return output_path
        
    elif file_type == '.pdf':
        # For PDFs, extract text and process row by row, only save cleaned .txt file
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(annotation_path)
            
            # Use row-based deidentification that preserves clinical impressions
            deidentified_text = deidentify_text_by_rows(text)
            
            # Save only the cleaned .txt file
            output_text_path = os.path.join(output_dir, "annotation_deidentified.txt")
            with open(output_text_path, 'w', encoding='utf-8') as file:
                file.write(deidentified_text)
            
            print(f"Processed annotation file saved to: {os.path.basename(output_text_path)}")
            return output_text_path
        except Exception as e:
            print(f"Failed to process PDF {annotation_path}: {str(e)}")
            return None
            
    elif file_type in ['.jpeg', '.jpg', '.png']:
        text = extract_text_from_image(annotation_path)
        
        # Use row-based deidentification that preserves clinical impressions
        deidentified_text = deidentify_text_by_rows(text)
        
        # Save as .txt file
        output_path = os.path.join(output_dir, "annotation_deidentified.txt")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(deidentified_text)
        
        print(f"Processed annotation file saved to: {output_path}")
        return output_path
        
    else:
        print(f"Unsupported annotation file type: {file_type}")
        return None




def intensity_normalization(image):
    """
    Apply intensity normalization to the image.
    """
    # Use SimpleITK's RescaleIntensityImageFilter
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMaximum(255)
    rescaler.SetOutputMinimum(0)
    return rescaler.Execute(image)


def gaussian_smoothing(image, sigma=1.0):
    """
    Apply Gaussian smoothing to the image.
    """
    # Use SimpleITK's SmoothingRecursiveGaussianImageFilter
    smoother = sitk.SmoothingRecursiveGaussianImageFilter()
    smoother.SetSigma(sigma)
    return smoother.Execute(image)


def main():
    """
    Main function to process all ZIP files in a directory.
    """
    parser = argparse.ArgumentParser(description='De-identify DICOM files in ZIP archives')
    parser.add_argument('directory', help='Directory containing ZIP files to process')
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.directory)
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist")
        return
    
    # Find all ZIP files in the directory
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    
    if not zip_files:
        print(f"No ZIP files found in directory '{input_dir}'")
        return
    
    print(f"Found {len(zip_files)} ZIP files to process")
    
    # Process each ZIP file with numeric naming
    for idx, zip_path in enumerate(zip_files, 1):
        zip_filename = os.path.basename(zip_path)
        print(f"\nProcessing {idx}/{len(zip_files)}: {zip_filename}")
        
        # Create output filename with numeric directory
        output_dir_name = f"{idx:03d}_DID"
        output_path = os.path.join(input_dir, output_dir_name + ".zip")
        
        try:
            process_zip(zip_path, output_path)
            print(f"✓ Successfully processed as {output_dir_name}.zip")
        except Exception as e:
            print(f"✗ Error processing {zip_filename}: {str(e)}")


def zip_directory(directory_path, output_path, compression=zipfile.ZIP_DEFLATED):
    """
    Compress a directory into a ZIP file.
    """
    # Convert to absolute paths
    directory_path = os.path.abspath(directory_path)

    # List to store included files
    included_files = []

    try:
        with zipfile.ZipFile(output_path, 'w', compression) as zipf:
            # Walk through directory
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # Get full file path
                    file_path = os.path.join(root, file)

                    # Get relative path for archive
                    relative_path = os.path.relpath(file_path, directory_path)

                    # Add file to ZIP
                    zipf.write(file_path, relative_path)
                    included_files.append(relative_path)

                    print(f"Added: {relative_path}")

        print(f"\nZIP file created successfully at: {output_path}")
        print(f"Total files: {len(included_files)}")

        return included_files

    except Exception as e:
        print(f"Error creating ZIP file: {str(e)}")
        return []


def process_zip(zip_path, output_path):
    """
    Process a zip file by extracting its contents, processing the DICOM files,
    and saving the results as a new zip file.
    """
    temp_dir = os.path.dirname(output_path)
    
    # Extract the zip file
    dicom_folder_paths, annotation_path = unzip_file(zip_path, os.path.join(temp_dir, "extracted"))
    
    if dicom_folder_paths:  # Check if list is not empty
        # Create output directory
        output_dir = output_path.rstrip('.zip')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Found {len(dicom_folder_paths)} DICOM directories to process")
        
        # Step 1: De-identify all DICOM files in each directory
        total_processed_files = 0
        processed_dicom_base_dir = os.path.join(output_dir, "dicom_files")
        
        for dicom_folder_path in dicom_folder_paths:
            # Get the folder name to preserve structure
            folder_name = os.path.basename(dicom_folder_path)
            print(f"\nProcessing DICOM directory: {dicom_folder_path}")
            
            # Create subdirectory in output for this DICOM folder
            processed_dicom_subdir = os.path.join(processed_dicom_base_dir, folder_name)
            processed_file_paths = remove_phi_metadata(dicom_folder_path, processed_dicom_subdir)
            total_processed_files += len(processed_file_paths)
            print(f"- Processed {len(processed_file_paths)} files from {folder_name}")
        
        # Step 2: Process the annotation file if it exists
        if annotation_path:
            print(f"\nProcessing annotation file: {annotation_path}")
            processed_annotation_path = process_annotation_file(annotation_path, output_dir)
            print(f"- Created processed annotation file: {processed_annotation_path}")
        
        print(f"\nProcessing complete:")
        print(f"- Processed {total_processed_files} DICOM files from {len(dicom_folder_paths)} directories")
        
        # Step 3: Create final zip file
        output_zip = output_path
        zip_directory(output_dir, output_zip)
        
        # Clean up temporary processing directory
        shutil.rmtree(output_dir)
        
        # Clean up extraction directory
        extraction_dir = os.path.join(temp_dir, "extracted")
        if os.path.exists(extraction_dir):
            shutil.rmtree(extraction_dir)
        
        print(f"Processing complete. Result saved to: {output_zip}")
    else:
        print(f"Failed to process {zip_path}: no DICOM files found")


if __name__ == "__main__":
    main()