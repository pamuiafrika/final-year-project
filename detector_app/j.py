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

class AdvancedPDFSteganography:
    """
    Advanced PDF Steganography Tool for hiding PNG images within PDF documents
    using multiple steganography techniques for maximum concealment.
    """

    def __init__(self, password=None):
        """
        Initialize the steganography tool with optional encryption password.

        Args:
            password (str): Optional password for encrypting the hidden data
        """
        self.password = password
        self.chunk_size = 1024  # Size of data chunks for distribution
        self.magic_marker = b"\x89PNG_HIDDEN\x0D\x0A\x1A\x0A"  # Custom marker for data identification

    def main(self, pdf_path, png_path, output_path):
        """
        Main function to hide PNG image within PDF document using advanced steganography.

        Args:
            pdf_path (str): Path to the input PDF document
            png_path (str): Path to the PNG image to hide
            output_path (str): Path for the output PDF with hidden image

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("[+] Starting advanced PDF steganography process...")

            # Step 1: Load and validate input files
            pdf_reader, png_data = self._load_and_validate_files(pdf_path, png_path)
            if not pdf_reader or not png_data:
                return False

            # Step 2: Prepare PNG data for hiding
            prepared_data = self._prepare_png_data(png_data)
            if not prepared_data:
                return False

            #Step 3: Encrypt the prepared data
            encrypted_data = self._encrypt_data(prepared_data)
            if not encrypted_data:
                return False

            # Step 4: Fragment data for distribution
            data_fragments = self._fragment_data(prepared_data)
            if not data_fragments:
                return False

            # Step 5: Hide data using steganography techniques
            modified_pdf = self._hide_data_multi_technique(pdf_reader, data_fragments)
            if not modified_pdf:
                return False

            # Step 6: Apply final obfuscation and save
            success = self._finalize_and_save(modified_pdf, output_path)

            if success:
                print(f"[+] Successfully hidden PNG image in PDF: {output_path}")
                if os.path.exists(pdf_path):
                    print(f"[+] Original PDF size: {os.path.getsize(pdf_path)} bytes")
                if os.path.exists(output_path):
                    print(f"[+] Modified PDF size: {os.path.getsize(output_path)} bytes")
                    if os.path.exists(pdf_path):
                        print(f"[+] Size increase: {os.path.getsize(output_path) - os.path.getsize(pdf_path)} bytes")

            return success

        except Exception as e:
            print(f"[-] Error in main steganography process: {str(e)}")
            return False

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
            if not os.path.exists(pdf_path) or not os.path.exists(png_path):
                print("[-] Input files not found")
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
                'timestamp': str(int(os.path.getmtime(__file__) if os.path.exists(__file__) else 0))
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

            print(f"[+] Prepared data: {len(png_data)} -> {len(prepared_data)} bytes (compression ratio: {len(prepared_data)/len(png_data):.2f})")

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
            # Derive key from password
            password_bytes = self.password.encode('utf-8')
            salt = b'pdf_stego_salt_2023'  # Fixed salt for consistency
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

    def _fragment_data(self, encrypted_data):
        """
        Fragment encrypted data into chunks for distributed hiding.

        Args:
            encrypted_data (bytes): Encrypted data to fragment

        Returns:
            list: List of data fragments with metadata
        """
        try:
            fragments = []
            total_chunks = (len(encrypted_data) + self.chunk_size - 1) // self.chunk_size

            for i in range(0, len(encrypted_data), self.chunk_size):
                chunk = encrypted_data[i:i + self.chunk_size]
                chunk_id = i // self.chunk_size

                # Add chunk metadata
                fragment = {
                    'id': chunk_id,
                    'total_chunks': total_chunks,
                    'size': len(chunk),
                    'data': chunk,
                    'checksum': hashlib.md5(chunk).hexdigest()
                }
                fragments.append(fragment)

            print(f"[+] Data fragmented into {len(fragments)} chunks")
            return fragments

        except Exception as e:
            print(f"[-] Error fragmenting data: {str(e)}")
            return []

    def _hide_data_multi_technique(self, pdf_reader, data_fragments):
        """
        Hide data fragments using multiple steganography techniques.

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

            # Technique 1: Metadata Hiding (most reliable)
            self._hide_in_metadata(pdf_writer, data_fragments)

            # Technique 2: Comment hiding (if metadata isn't enough)
            # if len(data_fragments) > 10:  # Only if we have many fragments
            #     self._hide_in_comments_and_markers(pdf_writer, data_fragments[:min(10, len(data_fragments))])

            print("[+] Data hiding completed using multiple techniques")
            return pdf_writer

        except Exception as e:
            print(f"[-] Error in multi-technique hiding: {str(e)}")
            return None

    def _hide_in_metadata(self, pdf_writer, fragments):
        """
        Hide data fragments in PDF metadata fields.

        Args:
            pdf_writer: PDF writer object
            fragments (list): Data fragments to hide
        """
        try:
            # Get existing metadata or create new
            metadata = pdf_writer.metadata if pdf_writer.metadata else {}
            if not isinstance(metadata, dict):
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

    def _hide_in_comments_and_markers(self, pdf_writer, fragments):
        """
        Hide data fragments in PDF comments and object markers.

        Args:
            pdf_writer: PDF writer object
            fragments (list): Data fragments to hide
        """
        try:
            if not pdf_writer.pages:
                print("[-] No pages available for comment hiding")
                return

            for i, fragment in enumerate(fragments):
                if i >= len(pdf_writer.pages):
                    break  # Don't exceed available pages

                # Encode fragment
                fragment_str = str(fragment).encode('utf-8')
                encoded_data = base64.b64encode(fragment_str).decode('utf-8')

                # Create annotation object with hidden data in comments
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
                page = pdf_writer.pages[i % len(pdf_writer.pages)]
                if "/Annots" not in page:
                    page["/Annots"] = ArrayObject()
                if not isinstance(page["/Annots"], ArrayObject):
                    page["/Annots"] = ArrayObject()
                page["/Annots"].append(annotation)

            print(f"[+] Hidden {min(len(fragments), len(pdf_writer.pages))} fragments in comments/markers")

        except Exception as e:
            print(f"[-] Error hiding in comments/markers: {str(e)}")

    def _finalize_and_save(self, pdf_writer, output_path):
        """
        Finalize the PDF and save to output path.

        Args:
            pdf_writer: Modified PDF writer object
            output_path (str): Path to save the output PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    def extract_hidden_png(self, pdf_path, output_png_path):
        """
        Extract hidden PNG image from a PDF document.

        Args:
            pdf_path (str): Path to PDF with hidden image
            output_png_path (str): Path to save extracted PNG

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("[+] Starting PNG extraction process...")

            # Load PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
            
            pdf_reader = PdfReader(io.BytesIO(pdf_data))

            # Extract data fragments from various locations
            fragments = []
            fragments.extend(self._extract_from_metadata(pdf_reader))
            fragments.extend(self._extract_from_comments(pdf_reader))

            if not fragments:
                print("[-] No hidden data found")
                return False

            # Reconstruct data
            encrypted_data = self._reconstruct_data(fragments)
            if not encrypted_data:
                return False

            # Decrypt data
            decrypted_data = self._decrypt_data(encrypted_data)
            if not decrypted_data:
                return False

            # Extract PNG from prepared data
            png_data = self._extract_png_from_prepared_data(decrypted_data)
            if not png_data:
                return False

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

            # Save PNG
            with open(output_png_path, 'wb') as png_file:
                png_file.write(png_data)

            print(f"[+] PNG successfully extracted to: {output_png_path}")
            return True

        except Exception as e:
            print(f"[-] Error extracting PNG: {str(e)}")
            return False

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
                            if "/Contents" in annot:
                                content = str(annot["/Contents"])
                                if "ProcessingData_" in content:
                                    try:
                                        # Extract encoded data after the prefix
                                        parts = content.split("ProcessingData_")[1]
                                        encoded_data = parts.split("_", 1)[1]  # Skip the number
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
        """Reconstruct encrypted data from fragments."""
        try:
            if not fragments:
                print("[-] No fragments to reconstruct")
                return None

            # Sort fragments by ID
            fragments.sort(key=lambda x: x['id'])

            # Verify completeness
            total_chunks = fragments[0]['total_chunks']
            if len(fragments) != total_chunks:
                print(f"[-] Missing fragments: expected {total_chunks}, found {len(fragments)}")
                return None

            # Verify sequence
            for i, fragment in enumerate(fragments):
                if fragment['id'] != i:
                    print(f"[-] Fragment sequence error at position {i}")
                    return None

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
            # Derive key from password
            password_bytes = self.password.encode('utf-8')
            salt = b'pdf_stego_salt_2023'
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

def main(pdf_path, png_path, output_path, password=None):
    """
    Main function to hide PNG image within PDF document.

    Args:
        pdf_path (str): Path to input PDF document
        png_path (str): Path to PNG image to hide
        output_path (str): Path for output PDF with hidden image
        password (str): Optional password for encryption

    Returns:
        bool: True if successful, False otherwise
    """
    stego_tool = AdvancedPDFSteganography(password)
    return stego_tool.main(pdf_path, png_path, output_path)

def extract_png(pdf_path, output_png_path, password=None):
    """
    Extract hidden PNG image from PDF document.

    Args:
        pdf_path (str): Path to PDF with hidden image
        output_png_path (str): Path to save extracted PNG
        password (str): Optional password for decryption

    Returns:
        bool: True if successful, False otherwise
    """
    stego_tool = AdvancedPDFSteganography(password)
    return stego_tool.extract_hidden_png(pdf_path, output_png_path)

# Example usage
if __name__ == "__main__":
    # Update these paths to match your file locations
    pdf_input = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/5G Network_ Security and Solutions_.pdf"
    png_input = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/media/stego/input_pngs/Screenshot_20250524_151908_LuMOERY.png"
    pdf_output = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/output_with_hidden_image.pdf"
    
    # Hide the PNG image
    print("=== HIDING PNG IN PDF ===")
    success = main(pdf_input, png_input, pdf_output, password="my_secret_password")
    
    if success:
        print("\n[+] Steganography process completed successfully!")
        print("The PDF appears normal but contains the hidden PNG image.")
        
        # Extract the hidden PNG (for verification)
        print("\n=== EXTRACTING HIDDEN PNG ===")
        extracted_png = "extracted_image.png"
        extract_success = extract_png(pdf_output, extracted_png, password="my_secret_password")
        
        if extract_success:
            print(f"[+] Hidden PNG successfully extracted to: {extracted_png}")
        else:
            print("[-] Failed to extract hidden PNG")
    else:
        print("[-] Steganography process failed")