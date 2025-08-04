# DID Tool - DICOM De-identification User Manual

## Overview
The DID Tool is a standalone executable that removes Protected Health Information (PHI) from DICOM medical imaging files and associated reports. It processes ZIP files containing DICOM images and PDF reports.

## System Requirements
- **Windows**: Windows 10 or later (64-bit)
- **macOS**: macOS 10.14 or later
- **Linux**: Most modern distributions
- **Storage**: At least 2GB free space for processing
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)

## Installation
1. Download the appropriate executable for your system:
   - Windows: `DID_Tool.exe`
   - macOS/Linux: `DID_Tool`
2. No additional installation required - it's a standalone executable

## Quick Start

### Step 1: Prepare Your Data
Create a ZIP file containing:
```
your_data.zip
├── Series1/           # DICOM image directory
│   ├── image001.dcm
│   ├── image002.dcm
│   └── ...
├── Series2/           # Additional DICOM directories (optional)
│   ├── image001.dcm
│   └── ...
└── report.pdf         # Medical report (can be in .txt/.pdf/.jpg formats)
```

### Step 2: Run the Tool

#### Option 1: Command Line (Recommended)
```bash
# Windows
DID_Tool.exe input_file.zip

# macOS/Linux
./DID_Tool input_file.zip
```

#### Option 2: Drag and Drop
- Drag your ZIP file onto the executable
- The tool will process automatically

### Step 3: Get Results
- Output file: `[input_number]_DID.zip`
- Example: `001_DID.zip` for first processed file
- Contains de-identified DICOM files and converted report text

## Command Line Options

```bash
# Basic usage
./DID_Tool input.zip

# Process multiple files
./DID_Tool file1.zip file2.zip file3.zip

# Specify output directory
./DID_Tool input.zip --output /path/to/output/

# Verbose output
./DID_Tool input.zip --verbose

# Help
./DID_Tool --help
```

## What Gets Removed (PHI Tags)
The tool removes these DICOM tags containing personal information:
- Patient Name
- Patient ID
- Patient Birth Date
- Patient Sex
- Patient Age
- Institution Name
- Institution Address
- Referring Physician Name
- Performing Physician Name
- Operator Name
- Study Date
- Study Time
- Series Date
- Series Time
- Acquisition Date
- Acquisition Time
- Content Date
- Content Time

## Processing Details

### DICOM Files
- **Input**: .dcm, .dicom, .ima files or files without extension
- **Processing**: PHI metadata removed, image data preserved
- **Output**: De-identified DICOM files in same directory structure

### PDF Reports
- **Input**: PDF files in ZIP root
- **Processing**: Text extracted and PHI patterns removed
- **Output**: De-identified text file (.txt)

### Expected Processing Time
- **Small dataset** (100-500 files): 30 seconds - 2 minutes
- **Medium dataset** (500-1000 files): 2-5 minutes
- **Large dataset** (1000+ files): 5+ minutes

## File Naming Convention
Output files follow this pattern:
- First file: `001_DID.zip`
- Second file: `002_DID.zip`
- And so on...

## Example Session
```
$ ./DID_Tool PATIENT_STUDY.zip

Starting DICOM De-identification Tool...
Processing: PATIENT_STUDY.zip

Extracting ZIP file...
Found DICOM directory: extracted/Series1
Found DICOM directory: extracted/Series2
Found annotation file: extracted/report.pdf

Processing DICOM directory: extracted/Series1
Processing 376 DICOM files...
Successfully processed 376 DICOM files

Processing DICOM directory: extracted/Series2  
Processing 376 DICOM files...
Successfully processed 376 DICOM files

Processing PDF annotation...
Converting PDF to de-identified text...

Creating output ZIP: 001_DID.zip
✓ Total files processed: 752 DICOM files
✓ PHI tags removed: 18 per file
✓ Output created: 001_DID.zip

Processing completed successfully!
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Permission Denied" Error
**Problem**: Cannot write to output directory
**Solutions**:
- Run as administrator (Windows) or with sudo (macOS/Linux)
- Choose a different output directory you have write access to
- Check if antivirus is blocking the executable

#### 2. "File Not Found" Error
**Problem**: Input ZIP file cannot be found
**Solutions**:
- Check file path is correct
- Use absolute path: `/full/path/to/file.zip`
- Ensure file exists and is accessible

#### 3. "Invalid ZIP File" Error
**Problem**: ZIP file is corrupted or invalid
**Solutions**:
- Test ZIP file with built-in extractor first
- Re-create the ZIP file
- Check file wasn't corrupted during transfer

#### 4. "No DICOM Files Found"
**Problem**: Tool cannot find DICOM files in ZIP
**Solutions**:
- Ensure DICOM files have correct extensions (.dcm, .dicom, .ima)
- Check directory structure matches expected format
- Verify DICOM files are valid using DICOM viewer

#### 5. "Memory Error" / Crashes
**Problem**: Not enough memory for large datasets
**Solutions**:
- Close other applications to free memory
- Process smaller batches of files
- Use computer with more RAM

#### 6. Antivirus False Positive
**Problem**: Antivirus flags executable as malicious
**Solutions**:
- Add executable to antivirus whitelist
- Temporarily disable real-time scanning
- Download from trusted source

#### 7. macOS "Cannot Open" Error
**Problem**: macOS blocks unsigned executable
**Solutions**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine DID_Tool

# Or run with override
sudo spctl --master-disable
```

#### 8. Linux Permission Error
**Problem**: Executable not permitted to run
**Solutions**:
```bash
# Make executable
chmod +x DID_Tool

# Run
./DID_Tool input.zip
```

### Getting Help

#### Check Version and Help
```bash
DID_Tool --version
DID_Tool --help
```

#### Enable Verbose Output
```bash
DID_Tool input.zip --verbose
```
This provides detailed processing information useful for troubleshooting.

#### Log Files
The tool creates log files in the same directory:
- `did_tool.log` - Processing details
- `error.log` - Error messages

## Data Privacy and Security

### Important Notes
- **Local Processing**: All processing happens locally on your computer
- **No Network Access**: Tool does not send data over the internet
- **Original Files**: Input files are never modified
- **Temporary Files**: Automatically cleaned up after processing

### Verification
After processing, verify de-identification by:
1. Opening output DICOM files in DICOM viewer
2. Checking that PHI fields are empty or anonymized
3. Reviewing converted text files for remaining PHI

## Technical Support

### Before Contacting Support
1. Check this troubleshooting section
2. Run with `--verbose` flag
3. Check log files for error details
4. Note your operating system and tool version

### When Reporting Issues Include
- Operating system and version
- Exact error message
- Input file characteristics (size, structure)
- Command used
- Log file contents

## Limitations
- **File Formats**: Only processes DICOM and PDF files
- **ZIP Only**: Input must be ZIP format
- **Text-based PHI**: May not catch all PHI in unusual formats
- **PDF Images**: Text in PDF images requires OCR (may not be 100% accurate)

---

**Version**: 1.0  
**Last Updated**: August 2025