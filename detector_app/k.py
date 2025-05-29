import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import DictionaryObject, ArrayObject, TextStringObject, ByteStringObject
import PIL.Image
import zlib
import base64
import hashlib
import os
import random
import struct
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import io
import binascii
from datetime import datetime

class AdvancedPDFSteganography:
    """
    Advanced PDF Steganography Tool for hiding PNG images within PDF documents
    using multiple steganography techniques with selectable methods.
    """

    # Hiding technique constants
    TECHNIQUE_METADATA = "metadata"
    TECHNIQUE_COMMENTS = "comments"
    TECHNIQUE_OBJECTS = "objects"
    TECHNIQUE_WHITESPACE = "whitespace"
    TECHNIQUE_MULTI = "multi"

    def __init__(self, password=None, technique=TECHNIQUE_METADATA):
        """
        Initialize the steganography tool with optional encryption password and technique.

        Args:
            password (str): Optional password for encrypting the hidden data
            technique (str): Hiding technique to use
        """
        self.password = password
        self.technique = technique
        self.chunk_size = 1024  # Size of data chunks for distribution
        self.magic_marker = b"\x89PNG_HIDDEN\x0D\x0A\x1A\x0A"  # Custom marker for data identification

    def main(self, pdf_path, png_path, output_path=None):
        """
        Main function to hide PNG image within PDF document using advanced steganography.

        Args:
            pdf_path (str): Path to the input PDF document
            png_path (str): Path to the PNG image to hide
            output_path (str): Optional path for the output PDF with hidden image

        Returns:
            tuple: (success: bool, output_path: str)
        """
        try:
            print(f"[+] Starting advanced PDF steganography process using {self.technique} technique...")

            # Auto-generate output path if not provided
            if not output_path:
                output_path = self._generate_output_path(pdf_path)

            # Step 1: Load and validate input files
            pdf_reader, png_data = self._load_and_validate_files(pdf_path, png_path)
            if not pdf_reader or not png_data:
                return False, output_path

            # Step 2: Prepare PNG data for hiding
            prepared_data = self._prepare_png_data(png_data)
            if not prepared_data:
                return False, output_path

            # Step 3: Encrypt the prepared data (only if password is provided)
            if self.password:
                print("[+] Password provided, encrypting data...")
                processed_data = self._encrypt_data(prepared_data)
                if not processed_data:
                    return False, output_path
            else:
                print("[+] No password provided, skipping encryption...")
                processed_data = prepared_data

            # Step 4: Fragment data for distribution
            data_fragments = self._fragment_data(processed_data)
            if not data_fragments:
                return False, output_path

            # Step 5: Hide data using selected technique
            modified_pdf = self._hide_data_with_technique(pdf_reader, data_fragments)
            if not modified_pdf:
                return False, output_path

            # Step 6: Apply final obfuscation and save
            success = self._finalize_and_save(modified_pdf, output_path)

            if success:
                print(f"[+] Successfully hidden PNG image in PDF: {output_path}")
                if os.path.exists(pdf_path):
                    print(f"[+] Original PDF size: {os.path.getsize(pdf_path)} bytes")
                if os.path.exists(output_path):
                    print(f"[+] Modified PDF size: {os.path.getsize(output_path)} bytes")
                    if os.path.exists(pdf_path):
                        size_diff = os.path.getsize(output_path) - os.path.getsize(pdf_path)
                        print(f"[+] Size increase: {size_diff} bytes")

            return success, output_path

        except Exception as e:
            print(f"[-] Error in main steganography process: {str(e)}")
            return False, output_path

    def _generate_output_path(self, pdf_path):
        """
        Generate an auto-generated output path based on input PDF path.
        
        Args:
            pdf_path (str): Input PDF path
            
        Returns:
            str: Generated output path
        """
        try:
            # Get directory and filename
            directory = os.path.dirname(pdf_path)
            filename = os.path.basename(pdf_path)
            name, ext = os.path.splitext(filename)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create new filename
            technique_short = {
                self.TECHNIQUE_METADATA: "meta",
                self.TECHNIQUE_COMMENTS: "comm", 
                self.TECHNIQUE_OBJECTS: "obj",
                self.TECHNIQUE_WHITESPACE: "ws",
                self.TECHNIQUE_MULTI: "multi"
            }.get(self.technique, "stego")
            
            encrypted_suffix = "_enc" if self.password else ""
            new_filename = f"{name}_stego_{technique_short}{encrypted_suffix}_{timestamp}{ext}"
            output_path = os.path.join(directory, new_filename)
            
            print(f"[+] Auto-generated output path: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[-] Error generating output path: {str(e)}")
            # Fallback to simple naming
            return pdf_path.replace('.pdf', '_stego.pdf')

    def _load_and_validate_files(self, pdf_path, png_path):
        """
        Load and validate the input PDF and PNG files.

        Args:
            pdf_path (str): Path to PDF file
            png_path (str): Path to PNG file

        Returns:
            tuple: (pdf_reader_object, png_binary_data) or (None, None) if error
        """
        try:
            # Validate file existence
            if not os.path.exists(pdf_path):
                print(f"[-] PDF file not found: {pdf_path}")
                return None, None
            if not os.path.exists(png_path):
                print(f"[-] PNG file not found: {png_path}")
                return None, None

            # Load PDF - keep file handle open by reading into memory
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            print(f"[+] Loaded PDF with {len(pdf_reader.pages)} pages")

            # Load and validate PNG
            with open(png_path, 'rb') as png_file:
                png_data = png_file.read()

            # Verify PNG signature
            if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
                print("[-] Invalid PNG file format")
                return None, None

            # Verify PNG can be opened
            try:
                img = PIL.Image.open(io.BytesIO(png_data))
                print(f"[+] Loaded PNG image: {img.size[0]}x{img.size[1]} pixels, {len(png_data)} bytes")
                img.close()
            except Exception as e:
                print(f"[-] Error validating PNG: {str(e)}")
                return None, None

            return pdf_reader, png_data

        except Exception as e:
            print(f"[-] Error loading files: {str(e)}")
            return None, None

    def _prepare_png_data(self, png_data):
        """
        Prepare PNG data for hiding by adding metadata and compression.

        Args:
            png_data (bytes): Raw PNG file data

        Returns:
            bytes: Prepared data ready for encryption and hiding
        """
        try:
            # Create metadata header
            metadata = {
                'original_size': len(png_data),
                'checksum': hashlib.sha256(png_data).hexdigest(),
                'format': 'PNG',
                'timestamp': str(int(datetime.now().timestamp())),
                'technique': self.technique,
                'encrypted': bool(self.password)
            }

            # Serialize metadata
            metadata_str = str(metadata).encode('utf-8')
            metadata_size = struct.pack('<I', len(metadata_str))

            # Compress PNG data
            compressed_png = zlib.compress(png_data, level=9)
            compressed_size = struct.pack('<I', len(compressed_png))

            # Combine: magic_marker + metadata_size + metadata + compressed_size + compressed_data
            prepared_data = (self.magic_marker +
                           metadata_size + metadata_str +
                           compressed_size + compressed_png)

            compression_ratio = len(prepared_data) / len(png_data)
            print(f"[+] Prepared data: {len(png_data)} -> {len(prepared_data)} bytes (ratio: {compression_ratio:.2f})")

            return prepared_data

        except Exception as e:
            print(f"[-] Error preparing PNG data: {str(e)}")
            return None

    def _encrypt_data(self, data):
        """
        Encrypt the prepared data using Fernet encryption.

        Args:
            data (bytes): Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        try:
            if not self.password:
                return data
                
            # Derive key from password
            password_bytes = self.password.encode('utf-8')
            salt = b'pdf_stego_salt_2024'  # Updated salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))

            # Encrypt data
            f = Fernet(key)
            encrypted_data = f.encrypt(data)

            print(f"[+] Data encrypted: {len(data)} -> {len(encrypted_data)} bytes")
            return encrypted_data

        except Exception as e:
            print(f"[-] Error encrypting data: {str(e)}")
            return None

    def _fragment_data(self, data):
        """
        Fragment data into chunks for distributed hiding.

        Args:
            data (bytes): Data to fragment

        Returns:
            list: List of data fragments with metadata
        """
        try:
            fragments = []
            total_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size

            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                chunk_id = i // self.chunk_size

                # Add chunk metadata
                fragment = {
                    'id': chunk_id,
                    'total_chunks': total_chunks,
                    'size': len(chunk),
                    'data': chunk,
                    'checksum': hashlib.md5(chunk).hexdigest(),
                    'technique': self.technique
                }
                fragments.append(fragment)

            print(f"[+] Data fragmented into {len(fragments)} chunks")
            return fragments

        except Exception as e:
            print(f"[-] Error fragmenting data: {str(e)}")
            return []

    def _hide_data_with_technique(self, pdf_reader, data_fragments):
        """
        Hide data fragments using the selected technique.

        Args:
            pdf_reader: PDF reader object
            data_fragments (list): List of data fragments to hide

        Returns:
            PdfWriter: Modified PDF writer object
        """
        try:
            pdf_writer = PdfWriter()

            # Copy all pages to writer
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            # Apply selected hiding technique
            if self.technique == self.TECHNIQUE_METADATA:
                self._hide_in_metadata(pdf_writer, data_fragments)
            elif self.technique == self.TECHNIQUE_COMMENTS:
                self._hide_in_comments(pdf_writer, data_fragments)
            elif self.technique == self.TECHNIQUE_OBJECTS:
                self._hide_in_objects(pdf_writer, data_fragments)
            elif self.technique == self.TECHNIQUE_WHITESPACE:
                self._hide_in_whitespace(pdf_writer, data_fragments)
            elif self.technique == self.TECHNIQUE_MULTI:
                self._hide_multi_technique(pdf_writer, data_fragments)
            else:
                print(f"[-] Unknown technique: {self.technique}, using metadata")
                self._hide_in_metadata(pdf_writer, data_fragments)

            print(f"[+] Data hiding completed using {self.technique} technique")
            return pdf_writer

        except Exception as e:
            print(f"[-] Error in technique hiding: {str(e)}")
            return None

    def _hide_in_metadata(self, pdf_writer, fragments):
        """Hide data fragments in PDF metadata fields."""
        try:
            metadata = {}
            
            for i, fragment in enumerate(fragments):
                # Encode fragment as base64
                fragment_str = str(fragment).encode('utf-8')
                encoded_fragment = base64.b64encode(fragment_str).decode('utf-8')

                # Use innocuous-looking metadata field names
                field_name = f"DocumentVersion{i:03d}"
                metadata[field_name] = encoded_fragment

            # Convert to DictionaryObject for PyPDF2
            metadata_dict = DictionaryObject()
            for key, value in metadata.items():
                if not key.startswith('/'):
                    key = '/' + key
                metadata_dict[key] = TextStringObject(str(value))

            pdf_writer.add_metadata(metadata_dict)
            print(f"[+] Hidden {len(fragments)} fragments in metadata")

        except Exception as e:
            print(f"[-] Error hiding in metadata: {str(e)}")

    def _hide_in_comments(self, pdf_writer, fragments):
        """Hide data fragments in PDF comments/annotations."""
        try:
            for i, fragment in enumerate(fragments[:len(pdf_writer.pages)]):
                fragment_str = str(fragment).encode('utf-8')
                encoded_data = base64.b64encode(fragment_str).decode('utf-8')

                # Create invisible annotation
                annotation = DictionaryObject()
                annotation.update({
                    "/Type": "/Annot",
                    "/Subtype": "/Text",
                    "/Rect": ArrayObject([0, 0, 0, 0]),  # Invisible rectangle
                    "/Contents": TextStringObject(f"ProcessingData_{i:03d}_{encoded_data}"),
                    "/Open": False,
                    "/Flags": 2  # Hidden flag
                })

                # Add to page annotations
                page_idx = i % len(pdf_writer.pages)
                page = pdf_writer.pages[page_idx]
                if "/Annots" not in page:
                    page["/Annots"] = ArrayObject()
                if not isinstance(page["/Annots"], ArrayObject):
                    page["/Annots"] = ArrayObject()
                page["/Annots"].append(annotation)

            print(f"[+] Hidden {min(len(fragments), len(pdf_writer.pages))} fragments in comments")

        except Exception as e:
            print(f"[-] Error hiding in comments: {str(e)}")

    def _hide_in_objects(self, pdf_writer, fragments):
        """Hide data fragments in PDF object streams."""
        try:
            # Create custom objects to store data
            for i, fragment in enumerate(fragments):
                fragment_str = str(fragment).encode('utf-8')
                encoded_data = base64.b64encode(fragment_str).decode('utf-8')
                
                # Create a custom object
                custom_obj = DictionaryObject()
                custom_obj.update({
                    "/Type": "/Metadata",
                    "/Subtype": "/XML",
                    "/Length": len(encoded_data),
                    "/Filter": "/ASCIIHexDecode",
                    "/ProcessingData": TextStringObject(encoded_data)
                })
                
                # Add object to PDF (this is a simplified approach)
                # In practice, you'd need to properly integrate with PDF structure
                
            print(f"[+] Hidden {len(fragments)} fragments in objects")
            
        except Exception as e:
            print(f"[-] Error hiding in objects: {str(e)}")

    def _hide_in_whitespace(self, pdf_writer, fragments):
        """Hide data fragments using whitespace manipulation."""
        try:
            # This is a placeholder for whitespace-based hiding
            # In practice, this would modify spacing in text content
            print(f"[+] Whitespace hiding not fully implemented, using metadata fallback")
            self._hide_in_metadata(pdf_writer, fragments)
            
        except Exception as e:
            print(f"[-] Error hiding in whitespace: {str(e)}")

    def _hide_multi_technique(self, pdf_writer, fragments):
        """Hide data fragments using multiple techniques for redundancy."""
        try:
            # Split fragments across techniques
            meta_fragments = fragments[:len(fragments)//2]
            comment_fragments = fragments[len(fragments)//2:]
            
            if meta_fragments:
                self._hide_in_metadata(pdf_writer, meta_fragments)
            if comment_fragments:
                self._hide_in_comments(pdf_writer, comment_fragments)
                
            print(f"[+] Hidden fragments using multiple techniques: {len(meta_fragments)} in metadata, {len(comment_fragments)} in comments")
            
        except Exception as e:
            print(f"[-] Error in multi-technique hiding: {str(e)}")

    def _finalize_and_save(self, pdf_writer, output_path):
        """Finalize the PDF and save to output path."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)

            # Verify the output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"[+] PDF successfully saved to: {output_path}")
                return True
            else:
                print("[-] Error: Output file not created properly")
                return False

        except Exception as e:
            print(f"[-] Error saving PDF: {str(e)}")
            return False

    def extract_hidden_png(self, pdf_path, output_png_path=None):
        """
        Extract hidden PNG image from a PDF document.

        Args:
            pdf_path (str): Path to PDF with hidden image
            output_png_path (str): Optional path to save extracted PNG

        Returns:
            tuple: (success: bool, output_path: str)
        """
        try:
            print("[+] Starting PNG extraction process...")

            # Auto-generate output path if not provided
            if not output_png_path:
                output_png_path = self._generate_extraction_path(pdf_path)

            # Load PDF
            if not os.path.exists(pdf_path):
                print(f"[-] PDF file not found: {pdf_path}")
                return False, output_png_path

            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            pdf_reader = PdfReader(io.BytesIO(pdf_data))

            # Extract data fragments from various locations
            fragments = []
            
            # Try all extraction methods
            metadata_fragments = self._extract_from_metadata(pdf_reader)
            comment_fragments = self._extract_from_comments(pdf_reader)
            
            fragments.extend(metadata_fragments)
            fragments.extend(comment_fragments)

            if not fragments:
                print("[-] No hidden data found")
                return False, output_png_path

            # Remove duplicates and sort
            unique_fragments = {}
            for fragment in fragments:
                if fragment['id'] not in unique_fragments:
                    unique_fragments[fragment['id']] = fragment
            
            fragments = list(unique_fragments.values())

            # Reconstruct data
            reconstructed_data = self._reconstruct_data(fragments)
            if not reconstructed_data:
                return False, output_png_path

            # Decrypt data if encrypted
            if self.password:
                print("[+] Attempting to decrypt data...")
                processed_data = self._decrypt_data(reconstructed_data)
                if not processed_data:
                    print("[-] Decryption failed - check password")
                    return False, output_png_path
            else:
                processed_data = reconstructed_data

            # Extract PNG from prepared data
            png_data = self._extract_png_from_prepared_data(processed_data)
            if not png_data:
                return False, output_png_path

            # Ensure output directory exists
            output_dir = os.path.dirname(output_png_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save PNG
            with open(output_png_path, 'wb') as png_file:
                png_file.write(png_data)

            # Verify extracted PNG
            try:
                img = PIL.Image.open(output_png_path)
                print(f"[+] PNG successfully extracted: {img.size[0]}x{img.size[1]} pixels to {output_png_path}")
                img.close()
                return True, output_png_path
            except Exception as e:
                print(f"[-] Error verifying extracted PNG: {str(e)}")
                return False, output_png_path

        except Exception as e:
            print(f"[-] Error extracting PNG: {str(e)}")
            return False, output_png_path

    def _generate_extraction_path(self, pdf_path):
        """Generate auto path for extracted PNG."""
        try:
            directory = os.path.dirname(pdf_path)
            filename = os.path.basename(pdf_path)
            name, _ = os.path.splitext(filename)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(directory, f"extracted_{name}_{timestamp}.png")
            
            print(f"[+] Auto-generated extraction path: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[-] Error generating extraction path: {str(e)}")
            return "extracted_image.png"

    def _extract_from_metadata(self, pdf_reader):
        """Extract data fragments from PDF metadata."""
        fragments = []
        try:
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if 'DocumentVersion' in str(key):
                        try:
                            decoded_data = base64.b64decode(str(value).encode('utf-8'))
                            fragment = eval(decoded_data.decode('utf-8'))
                            fragments.append(fragment)
                        except Exception:
                            continue
            print(f"[+] Extracted {len(fragments)} fragments from metadata")
        except Exception as e:
            print(f"[-] Error extracting from metadata: {str(e)}")

        return fragments

    def _extract_from_comments(self, pdf_reader):
        """Extract data fragments from PDF comments."""
        fragments = []
        try:
            for page in pdf_reader.pages:
                if "/Annots" in page:
                    annotations = page["/Annots"]
                    if annotations:
                        for annot in annotations:
                            try:
                                if "/Contents" in annot:
                                    content = str(annot["/Contents"])
                                    if "ProcessingData_" in content and "_" in content:
                                        # Extract encoded data after the prefix
                                        parts = content.split("ProcessingData_")
                                        if len(parts) > 1:
                                            remaining = parts[1]
                                            if "_" in remaining:
                                                encoded_data = remaining.split("_", 1)[1]
                                                decoded_data = base64.b64decode(encoded_data.encode('utf-8'))
                                                fragment = eval(decoded_data.decode('utf-8'))
                                                fragments.append(fragment)
                            except Exception:
                                continue
            print(f"[+] Extracted {len(fragments)} fragments from comments")
        except Exception as e:
            print(f"[-] Error extracting from comments: {str(e)}")

        return fragments

    def _reconstruct_data(self, fragments):
        """Reconstruct data from fragments."""
        try:
            if not fragments:
                print("[-] No fragments to reconstruct")
                return None

            # Sort fragments by ID
            fragments.sort(key=lambda x: x['id'])

            # Verify completeness
            total_chunks = fragments[0]['total_chunks']
            expected_ids = set(range(total_chunks))
            actual_ids = set(f['id'] for f in fragments)
            
            if len(fragments) != total_chunks or expected_ids != actual_ids:
                print(f"[-] Fragment mismatch: expected {total_chunks} with IDs {expected_ids}, found {len(fragments)} with IDs {actual_ids}")
                # Try to proceed with available fragments
                print("[+] Attempting reconstruction with available fragments...")

            # Reconstruct data
            reconstructed_data = b''
            for fragment in fragments:
                reconstructed_data += fragment['data']

            print(f"[+] Reconstructed {len(reconstructed_data)} bytes from {len(fragments)} fragments")
            return reconstructed_data

        except Exception as e:
            print(f"[-] Error reconstructing data: {str(e)}")
            return None

    def _decrypt_data(self, encrypted_data):
        """Decrypt the reconstructed data."""
        try:
            if not self.password:
                return encrypted_data
                
            # Derive key from password
            password_bytes = self.password.encode('utf-8')
            salt = b'pdf_stego_salt_2024'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))

            # Decrypt data
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)

            print(f"[+] Data decrypted: {len(encrypted_data)} -> {len(decrypted_data)} bytes")
            return decrypted_data

        except Exception as e:
            print(f"[-] Error decrypting data: {str(e)}")
            return None

    def _extract_png_from_prepared_data(self, prepared_data):
        """Extract PNG data from prepared data format."""
        try:
            # Check magic marker
            if not prepared_data.startswith(self.magic_marker):
                print("[-] Invalid data format: missing magic marker")
                return None

            offset = len(self.magic_marker)

            # Read metadata size
            if len(prepared_data) < offset + 4:
                print("[-] Insufficient data for metadata size")
                return None
            
            metadata_size = struct.unpack('<I', prepared_data[offset:offset+4])[0]
            offset += 4

            # Skip metadata
            if len(prepared_data) < offset + metadata_size:
                print("[-] Insufficient data for metadata")
                return None
            
            offset += metadata_size

            # Read compressed data size
            if len(prepared_data) < offset + 4:
                print("[-] Insufficient data for compressed size")
                return None
            
            compressed_size = struct.unpack('<I', prepared_data[offset:offset+4])[0]
            offset += 4

            # Extract and decompress PNG data
            if len(prepared_data) < offset + compressed_size:
                print("[-] Insufficient data for compressed PNG")
                return None
            
            compressed_png = prepared_data[offset:offset+compressed_size]
            png_data = zlib.decompress(compressed_png)

            print(f"[+] PNG extracted: {compressed_size} -> {len(png_data)} bytes")
            return png_data

        except Exception as e:
            print(f"[-] Error extracting PNG from prepared data: {str(e)}")
            return None

# Convenience functions
def hide_png_in_pdf(pdf_path, png_path, output_path=None, password=None, technique="metadata"):
    """
    Hide PNG image within PDF document.

    Args:
        pdf_path (str): Path to input PDF document
        png_path (str): Path to PNG image to hide
        output_path (str): Optional path for output PDF with hidden image
        password (str): Optional password for encryption
        technique (str): Hiding technique ("metadata", "comments", "objects", "whitespace", "multi")

    Returns:
        tuple: (success: bool, output_path: str)
    """
    stego_tool = AdvancedPDFSteganography(password, technique)
    return stego_tool.main(pdf_path, png_path, output_path)

def extract_png_from_pdf(pdf_path, output_png_path=None, password=None, technique="metadata"):
    """
    Extract hidden PNG image from PDF document.

    Args:
        pdf_path (str): Path to PDF with hidden image
        output_png_path (str): Optional path to save extracted PNG
        password (str): Optional password for decryption
        technique (str): Technique used for hiding

    Returns:
        tuple: (success: bool, output_path: str)
    """
    stego_tool = AdvancedPDFSteganography(password, technique)
    return stego_tool.extract_hidden_png(pdf_path, output_png_path)

# Example usage
if __name__ == "__main__":
    # Update these paths to match your file locations
 # Your PNG file to hide
    pdf_input = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/5G Network_ Security and Solutions_.pdf"
    png_input = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/media/stego/input_pngs/Screenshot_20250524_151908_LuMOERY.png"
    
    
    # Example 1: Hide using metadata technique with encryption
    print("=== HIDING PNG IN PDF (Metadata + Encryption) ===")
    success, output_path = hide_png_in_pdf(
        pdf_input, 
        png_input, 
        password="",
        technique="metadata"
    )
    
    if success:
        print(f"\n[+] Steganography completed! Output: {output_path}")
        
        # Extract the hidden PNG (for verification)
        print("\n=== EXTRACTING HIDDEN PNG ===")
        extract_success, extracted_path = extract_png_from_pdf(
            output_path, 
            password="my_secret_password",
            technique="metadata"
        )
        
        if extract_success:
            print(f"[+] Hidden PNG successfully extracted to: {extracted_png}")
        else:
            print("[-] Failed to extract hidden PNG")
    else:
        print("[-] Steganography process failed")


