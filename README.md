# Azure Document Intelligence Output Parser

A comprehensive document analysis system that processes Azure Document Intelligence JSON files to extract personal names and classify sensitive content using advanced AI models.

## 🚀 Features

- **Document Analysis**: Processes Azure DI JSON files with advanced segmentation
- **AI-Powered Classification**: Uses Gemini Pro 2.5 for intelligent content analysis
- **Page Number Tracking**: Tracks original page numbers for each classification
- **CV/Resume Merging**: Automatically merges multi-page CVs into single classifications
- **Professional Reports**: Generates formatted Word documents and Markdown reports
- **Caching Support**: Implements prompt caching for cost optimization
- **Batch Processing**: Handles multiple documents efficiently

## 📋 Classification Categories

The system classifies content into these categories:

### Personal Information
- **1.1** Personal Information
- **1.2** Governors'/Executive Directors' Communications
- **1.3** Ethics Committee Materials
- **1.4** Attorney–Client Privilege
- **1.5** Security & Safety Information
- **1.6** Restricted Investigative Info
- **1.7** Confidential Third-Party Information
- **1.8** Corporate Administrative Matters
- **1.9** Financial Information

### Document Types
- **2.1** CV or Resume Content
- **2.2** Derogatory or Offensive Language

### Special Categories
- **3.1** Documents from Specific Entities (IFC, MIGA, INT, IMF)
- **3.2** Joint WBG Documents
- **3.3** Security-Marked Documents
- **3.4** Procurement Content

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Cloud credentials for Vertex AI
- Required Python packages (see requirements below)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Minaekramnia/Azure-DI-document-parser.git
cd Azure-DI-document-parser
```

2. Install dependencies:
```bash
pip install google-genai python-docx python-dotenv
```

3. Set up your environment variables:
```bash
# Create .env file with your Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## 📖 Usage

## **Current Final Pipeline (`end_to_end_pipeline_6.py`)**

Use `end_to_end_pipeline_6.py` for the latest production-style flow.

### **Run**

```bash
python end_to_end_pipeline_6.py
```

### **What It Does**

- Reads Azure DI JSON files from `INPUT_DIR`
- Separates each PDF into internal document units (segments)
- Classifies each unit with Gemini
- Produces file-level output plus unit-level details

### **Document Separation Rules (Current)**

1. **Continuation first**: if page-number sequence continues and font profile is similar, keep same document unit.
2. **Dramatic size change**: only large page-size changes trigger a new document.
3. **Title signal**: new title near page start can start a new document when continuation is not confirmed.

### **Final File-Level Classification Behavior**

- The file-level category is **all unique non-general categories concatenated** (not just top-1).
- If no sensitive class is found, result is `General / No Sensitive Category`.

### **JSON Fields Added for Highlighting**

- `classification_summary.unique_categories`
- `classification_summary.concatenated_categories`
- `classification_summary.category_pages`
- `classification_summary.category_page_ranges`
- `classification_summary.category_reasons`

These are designed to support PDF top-banner highlighting (category + page ranges + short reasons).

### Main Processing Script

The primary script for document analysis:

```bash
python Azure_DI_output_parser_WORKING_PageNumbers.py
```

**Features:**
- Processes Azure DI JSON files
- Extracts personal names
- Classifies sensitive content
- Includes page number tracking
- Merges multi-page CVs
- Outputs structured JSON results

### Output Structure

The system generates JSON files with this structure:

```json
{
  "document_path": "path/to/document.pdf.json",
  "total_pages": 5,
  "total_segments": 2,
  "segments": [
    {
      "segment_id": "segment_1",
      "pages": [1, 2],
      "page_range": "1-2",
      "extracted_names": ["John Doe", "Jane Smith"],
      "classifications": [
        {
          "category": "2.1 CV or Resume Content",
          "text": "Professional experience and education...",
          "bounding_box": [0, 0, 10, 12],
          "page_number": 1,
          "confidence_score": 0.95,
          "reason": "Contains complete CV information"
        }
      ]
    }
  ]
}
```

## 📄 Report Generation

### Convert to Word Documents

Generate professional Word reports:

```bash
python convert_to_word_fixed.py
```

**Features:**
- Formal document styling
- Cambria font throughout
- Professional formatting
- Summary statistics
- Classification breakdown

### Remove Names from Reports

Create clean versions without extracted names:

```bash
python remove_names_from_word_FIXED.py
```

### Final Formatting

Apply final formatting to reports:

```bash
python format_final_word_documents_FINAL.py
```

## 📁 Output Folders

The system creates organized output folders:

```
PI/
├── markdown_reports/          # Markdown versions of reports
├── word_reports/             # Original Word documents
├── final_word_reports/       # Cleaned Word documents (no names)
└── *_WORKING_PageNumbers_analysis.json  # Analysis results
```

## ⚙️ Configuration

### Prompt File
The system uses `Executive_Prompt.md` for AI analysis instructions. This file contains:
- Detailed classification rules
- Output format requirements
- Sensitivity guidelines

### Caching
The system implements intelligent caching:
- **First file**: Full prompt sent (~2000+ tokens)
- **Subsequent files**: Cached prompt (~100 tokens)
- **Cost savings**: Significant reduction in API costs

## 🔧 Advanced Features

### Document Segmentation
- **Size-based**: Detects document boundaries by page size changes
- **Title-based**: Identifies new documents by title presence
- **Page sequence**: Uses page numbering to merge related pages

### CV Processing
- Automatically detects multi-page CVs
- Merges related CV sections into single classification
- Maintains page number tracking across merged content

### Error Handling
- Graceful fallback for caching failures
- Comprehensive logging and validation
- Robust JSON parsing with error recovery

## 📊 Performance

- **Batch processing**: Handles multiple documents efficiently
- **Caching optimization**: Reduces API costs by ~90%
- **Memory efficient**: Processes large documents without memory issues
- **Fast execution**: Optimized for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `/docs` folder
- Review the example outputs in the repository

## 🎯 Use Cases

This system is designed for:
- **Compliance teams** processing sensitive documents
- **Legal departments** reviewing confidential materials
- **HR teams** analyzing CVs and personal information
- **Security teams** classifying sensitive content
- **Research organizations** processing large document collections

## 📈 Recent Updates

-  Added page number tracking for all classifications
-  Implemented CV merging for multi-page documents
-  Enhanced Word document formatting
-  Added comprehensive error handling
-  Optimized caching for cost reduction
-  Created professional report templates

---

