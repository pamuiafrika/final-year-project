"""
Core steganography logic for embedding and extracting PNG images in PDF documents.
"""

import base64
import os
import random
import re
import zlib
from io import BytesIO
from pikepdf import Pdf, PdfImage, Name, String, Array, Dictionary
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image

class PDFSteganography:
    """Class containing steganography methods for PDF documents"""
    
    def __init__(self):
        """Initialize the steganography engine"""
        self.methods = {
            'stream': self.object_stream_method,
            'metadata': self.metadata_method,
            'whitespace': self.whitespace_method,
            'components': self.document_components_method,
            'javascript': self.javascript_method
        }
        self.extract_methods = {
            'stream': self.extract_object_stream,
            'metadata': self.extract_metadata,
            'whitespace': self.extract_whitespace,
            'components': self.extract_document_components,
            'javascript': self.extract_javascript
        }
    
    def get_random_method(self):
        """Select a random steganography method"""
        return random.choice(list(self.methods.keys()))
    
    def encode_png(self, png_path):
        """Encode PNG file to base64"""
        with open(png_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def decode_png(self, encoded_string, output_path):
        """Decode base64 to PNG file"""
        try:
            img_data = base64.b64decode(encoded_string)
            with open(output_path, 'wb') as f:
                f.write(img_data)
            return True
        except Exception as e:
            print(f"Error decoding PNG: {e}")
            return False
    
    def verify_compatibility(self, method, input_pdf_path, input_png_path=None):
        """
        Verify that the chosen method is compatible with the provided files
        Returns (is_compatible, message)
        """
        # Basic checks
        if not os.path.exists(input_pdf_path):
            return False, "PDF file does not exist"
        
        if input_png_path and not os.path.exists(input_png_path):
            return False, "PNG file does not exist"
        
        # Method-specific checks
        try:
            # Check if PDF is readable
            with Pdf.open(input_pdf_path) as pdf:
                # Check method-specific requirements
                if method == 'metadata':
                    # Some PDFs have limitations on metadata modifications
                    try:
                        if not pdf.docinfo:
                            pdf.docinfo = {}
                        pdf.docinfo['/TestKey'] = 'TestValue'
                    except Exception:
                        return False, "This PDF doesn't support metadata modification"
                
                elif method == 'javascript':
                    # Check if PDF has JavaScript restrictions
                    if '/Names' in pdf.Root and '/JavaScript' in pdf.Root.Names:
                        # Need to validate we can modify existing JavaScript
                        try:
                            names_array = pdf.Root.Names.JavaScript.Names
                            # This just tests if we can read it
                            test = len(names_array)
                        except Exception:
                            return False, "Cannot modify JavaScript in this PDF"
                
                elif method == 'components':
                    # Check if we can add annotations
                    try:
                        reader = PdfReader(input_pdf_path)
                        writer = PdfWriter()
                        writer.add_page(reader.pages[0])
                        
                        # Try to add a test annotation
                        writer.add_annotation(
                            page_number=0,
                            annotation={
                                '/Subtype': '/Text',
                                '/Contents': 'Test',
                                '/Rect': [0, 0, 1, 1]
                            }
                        )
                    except Exception:
                        return False, "Cannot add annotations to this PDF"
            
            # If PNG is provided, check size compatibility
            if input_png_path:
                # Check if PNG size is reasonable for the method
                png_size = os.path.getsize(input_png_path)
                pdf_size = os.path.getsize(input_pdf_path)
                
                # For each method, establish rough size limitations
                if method == 'metadata' and png_size > 1 * 1024 * 1024:  # 1MB
                    return False, "PNG too large for metadata method (max 1MB)"
                
                if method == 'whitespace' and png_size > 0.5 * 1024 * 1024:  # 0.5MB
                    return False, "PNG too large for whitespace method (max 500KB)"
                
                if png_size > 5 * pdf_size:
                    return False, "PNG file too large compared to PDF size"
            
            return True, "Compatible"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"
    
    # Method 1: Object Stream Manipulation
    def object_stream_method(self, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF object streams"""
        encoded_png = self.encode_png(png_path)
        
        # Create a marker to identify our hidden data
        marker = "HIDDEN_PNG_DATA:"
        
        with Pdf.open(input_pdf) as pdf:
            # Create a new object to store our data
            hidden_obj = Dictionary({
                Name.Type: Name.Metadata,
                Name.Subtype: Name.XML,
                Name.STEG: String(f"{marker}{encoded_png}")
            })
            
            # Add the object to the PDF
            pdf.Root.STEG_Container = hidden_obj
            
            # Save the modified PDF
            pdf.save(output_pdf)
            
        return True

    def extract_object_stream(self, input_pdf, output_png):
        """Extract PNG from PDF object streams"""
        marker = "HIDDEN_PNG_DATA:"
        
        with Pdf.open(input_pdf) as pdf:
            if Name.STEG_Container in pdf.Root:
                hidden_obj = pdf.Root.STEG_Container
                if Name.STEG in hidden_obj:
                    data = str(hidden_obj.STEG)
                    if marker in data:
                        encoded_data = data.split(marker)[1]
                        if self.decode_png(encoded_data, output_png):
                            return True
            
            return False

    # Method 2: Metadata Embedding
    def metadata_method(self, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF metadata"""
        encoded_png = self.encode_png(png_path)
        
        # Split the data into manageable chunks to avoid metadata size limitations
        chunk_size = 1000
        chunks = [encoded_png[i:i+chunk_size] for i in range(0, len(encoded_png), chunk_size)]
        
        with Pdf.open(input_pdf) as pdf:
            # Create metadata if it doesn't exist
            if not pdf.docinfo:
                pdf.docinfo = {}
                
            # Store number of chunks for extraction
            pdf.docinfo['/StegChunks'] = str(len(chunks))
            
            # Store each chunk in a custom metadata field
            for i, chunk in enumerate(chunks):
                pdf.docinfo[f'/StegData{i}'] = chunk
                
            pdf.save(output_pdf)
            
        return True

    def extract_metadata(self, input_pdf, output_png):
        """Extract PNG from PDF metadata"""
        with Pdf.open(input_pdf) as pdf:
            if '/StegChunks' not in pdf.docinfo:
                return False
                
            chunks_count = int(pdf.docinfo['/StegChunks'])
            encoded_data = ""
            
            # Reassemble chunks
            for i in range(chunks_count):
                if f'/StegData{i}' in pdf.docinfo:
                    encoded_data += str(pdf.docinfo[f'/StegData{i}'])
            
            if self.decode_png(encoded_data, output_png):
                return True
            return False

    # Method 3: White Space and Comment Exploitation
    def whitespace_method(self, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF whitespace and comments"""
        encoded_png = self.encode_png(png_path)
        
        # Read the PDF as text
        with open(input_pdf, 'rb') as file:
            pdf_content = file.read()
        
        # Find a suitable location for injection
        # Look for "endobj" markers which often have whitespace after them
        pdf_text = pdf_content.decode('latin-1', errors='ignore')
        
        # Create a comment with our data
        marker = "%STEG_DATA"
        hidden_data = f"\n{marker}{encoded_png}{marker}\n"
        
        # Insert after the first endobj
        position = pdf_text.find("endobj")
        if position == -1:
            return False
        
        modified_content = (pdf_text[:position+6] + hidden_data + pdf_text[position+6:]).encode('latin-1')
        
        # Write the modified content
        with open(output_pdf, 'wb') as file:
            file.write(modified_content)
            
        return True

    def extract_whitespace(self, input_pdf, output_png):
        """Extract PNG from PDF whitespace and comments"""
        marker = "%STEG_DATA"
        
        # Read the PDF as text
        with open(input_pdf, 'rb') as file:
            pdf_content = file.read().decode('latin-1', errors='ignore')
        
        # Find our markers
        start = pdf_content.find(marker)
        if start == -1:
            return False
        
        end = pdf_content.find(marker, start + 1)
        if end == -1:
            return False
        
        encoded_data = pdf_content[start + len(marker):end]
        
        if self.decode_png(encoded_data, output_png):
            return True
        return False

    # Method 4: Document Components Manipulation
    def document_components_method(self, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF document components like annotations"""
        encoded_png = self.encode_png(png_path)
        
        # Split the data to avoid size limitations
        chunk_size = 2000
        chunks = [encoded_png[i:i+chunk_size] for i in range(0, len(encoded_png), chunk_size)]
        
        # Create a reader and writer
        reader = PdfReader(input_pdf)
        writer = PdfWriter()
        
        # Copy all pages
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            writer.add_page(page)
            
            # Create hidden annotations for each chunk
            for j, chunk in enumerate(chunks):
                writer.add_annotation(
                    page_number=i,
                    annotation={
                        '/Subtype': '/Text',
                        '/T': f'STEG_CHUNK_{j}',
                        '/Contents': chunk,
                        '/C': [0, 0, 0],  # Color
                        '/F': 4,  # Hidden flag
                        '/Rect': [0, 0, 1, 1]  # Small rectangle outside visible area
                    }
                )
        
        # Add the number of chunks as document information
        writer.add_metadata({'/StegChunksCount': str(len(chunks))})
        
        # Write the output
        with open(output_pdf, 'wb') as f:
            writer.write(f)
            
        return True

    def extract_document_components(self, input_pdf, output_png):
        """Extract PNG from PDF document components"""
        reader = PdfReader(input_pdf)
        
        if '/StegChunksCount' not in reader.metadata:
            return False
        
        chunks_count = int(reader.metadata.get('/StegChunksCount', '0'))
        chunks = {}
        
        # Extract data from annotations
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            if '/Annots' in page and page['/Annots']:
                for annot in page['/Annots']:
                    annot_obj = annot.get_object()
                    if '/T' in annot_obj and '/Contents' in annot_obj:
                        title = annot_obj['/T']
                        if title.startswith('STEG_CHUNK_'):
                            chunk_id = int(title.split('_')[-1])
                            chunks[chunk_id] = annot_obj['/Contents']
        
        # Combine chunks in order
        encoded_data = ""
        for i in range(chunks_count):
            if i in chunks:
                encoded_data += chunks[i]
        
        if self.decode_png(encoded_data, output_png):
            return True
        return False

    # Method 5: JavaScript Integration
    def javascript_method(self, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF JavaScript objects"""
        encoded_png = self.encode_png(png_path)
        
        with Pdf.open(input_pdf) as pdf:
            # Create JavaScript action with hidden data
            js_code = f"""
            var steg_data = "{encoded_png}";
            // This function appears to do something innocent
            function checkFormData() {{
                return true; // Hidden data is stored in steg_data variable
            }}
            """
            
            js_action = Dictionary({
                Name.S: Name.JavaScript,
                Name.JS: String(js_code)
            })
            
            # Add to document actions
            if Name.Names not in pdf.Root:
                pdf.Root.Names = Dictionary({})
            
            if Name.JavaScript not in pdf.Root.Names:
                pdf.Root.Names.JavaScript = Dictionary({})
                pdf.Root.Names.JavaScript.Names = Array([String("StegScript"), js_action])
            else:
                # Add to existing JavaScript names
                names_array = pdf.Root.Names.JavaScript.Names
                names_array.append(String("StegScript"))
                names_array.append(js_action)
            
            pdf.save(output_pdf)
            
        return True

    def extract_javascript(self, input_pdf, output_png):
        """Extract PNG from PDF JavaScript objects"""
        with Pdf.open(input_pdf) as pdf:
            if Name.Names not in pdf.Root or Name.JavaScript not in pdf.Root.Names:
                return False
            
            js_names = pdf.Root.Names.JavaScript.Names
            encoded_data = None
            
            # Look for our script
            for i in range(0, len(js_names), 2):
                if i+1 < len(js_names) and str(js_names[i]) == "StegScript":
                    js_action = js_names[i+1]
                    js_code = str(js_action.JS)
                    
                    # Extract the data using regex
                    match = re.search(r'var steg_data = "([^"]+)";', js_code)
                    if match:
                        encoded_data = match.group(1)
                        break
            
            if encoded_data and self.decode_png(encoded_data, output_png):
                return True
            
            return False
    
    def hide(self, method, input_pdf, output_pdf, png_path):
        """Hide PNG in PDF using specified method"""
        if method == 'random':
            method = self.get_random_method()
            
        if method not in self.methods:
            return False, f"Unknown method '{method}'"
        
        # Verify compatibility
        compatible, message = self.verify_compatibility(method, input_pdf, png_path)
        if not compatible:
            return False, message
            
        try:
            result = self.methods[method](input_pdf, output_pdf, png_path)
            return result, method
        except Exception as e:
            return False, str(e)

    def extract(self, method, input_pdf, output_png):
        """Extract PNG from PDF using specified method"""
        if method == 'random':
            # Try all methods
            for m in self.extract_methods:
                try:
                    if self.extract_methods[m](input_pdf, output_png):
                        return True, m
                except:
                    pass
            return False, "Failed to extract with any method"
            
        if method not in self.extract_methods:
            return False, f"Unknown method '{method}'"
            
        try:
            result = self.extract_methods[method](input_pdf, output_png)
            return result, method
        except Exception as e:
            return False, str(e)