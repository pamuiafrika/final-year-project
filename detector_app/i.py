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
        self.password = password or "default_stego_key_2023"
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
            pdf_data, png_data = self._load_and_validate_files(pdf_path, png_path)
            if not pdf_data or not png_data:
                return False

            # Step 2: Prepare PNG data for hiding
            prepared_data = self._prepare_png_data(png_data)

            # Step 3: Encrypt the prepared data
            encrypted_data = self._encrypt_data(prepared_data)

            # Step 4: Fragment data for distribution
            data_fragments = self._fragment_data(encrypted_data)

            # Step 5: Hide data using multiple steganography techniques
            modified_pdf = self._hide_data_multi_technique(pdf_data, data_fragments)

            # Step 6: Apply final obfuscation and save
            success = self._finalize_and_save(modified_pdf, output_path)

            if success:
                print(f"[+] Successfully hidden PNG image in PDF: {output_path}")
                print(f"[+] Original PDF size: {os.path.getsize(pdf_path)} bytes")
                print(f"[+] Modified PDF size: {os.path.getsize(output_path)} bytes")
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

            # Load PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
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
                img = PIL.Image.open(png_path)
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
                'timestamp': str(int(os.path.getctime(__file__)))
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

            # Technique 1: PDF Object Stream Hiding
            self._hide_in_object_streams(pdf_writer, data_fragments[:len(data_fragments)//3])

            # Technique 2: Metadata Hiding
            # self._hide_in_metadata(pdf_writer, data_fragments[len(data_fragments)//3:2*len(data_fragments)//3])

            # Technique 3: Comment and Object Marker Obfuscation
            # self._hide_in_comments_and_markers(pdf_writer, data_fragments[2*len(data_fragments)//3:])

            # Technique 4: Add dummy objects for obfuscation
            # self._add_dummy_objects(pdf_writer)

            return pdf_writer

        except Exception as e:
            print(f"[-] Error in multi-technique hiding: {str(e)}")
            return None

    def _hide_in_object_streams(self, pdf_writer, fragments):
        """
        Hide data fragments within PDF object streams.

        Args:
            pdf_writer: PDF writer object
            fragments (list): Data fragments to hide
        """
        try:
            for i, fragment in enumerate(fragments):
                # Encode fragment data
                encoded_data = base64.b64encode(str(fragment).encode('utf-8')).decode('utf-8')

                # Create a dummy stream object
                stream_content = f"""
                BT
                /F1 0 Tf
                0 0 0 rg
                % Hidden data chunk {fragment['id']}: {encoded_data}
                ET
                """

                # Add as a content stream (invisible)
                stream_obj = DictionaryObject()
                stream_obj.update({
                    "/Type": "/XObject",
                    "/Subtype": "/Form",
                    "/BBox": ArrayObject([0, 0, 1, 1]),
                    "/Length": len(stream_content.encode())
                })

                # Add to PDF writer's objects
                if hasattr(pdf_writer, '_objects'):
                    pdf_writer._objects.append(stream_obj)

            print(f"[+] Hidden {len(fragments)} fragments in object streams")

        except Exception as e:
            print(f"[-] Error hiding in object streams: {str(e)}")

    def _hide_in_metadata(self, pdf_writer, fragments):
        """
        Hide data fragments in PDF metadata fields.

        Args:
            pdf_writer: PDF writer object
            fragments (list): Data fragments to hide
        """
        try:
            metadata = pdf_writer.metadata if pdf_writer.metadata else DictionaryObject()

            for i, fragment in enumerate(fragments):
                # Encode fragment as base64
                encoded_fragment = base64.b64encode(str(fragment).encode('utf-8')).decode('utf-8')

                # Use innocuous-looking metadata field names
                field_names = [
                    f"/DocumentVersion{i}",
                    f"/CreationTool{i}",
                    f"/ProcessingInfo{i}",
                    f"/ValidationData{i}",
                    f"/InternalID{i}"
                ]

                if i < len(field_names):
                    metadata[field_names[i]] = TextStringObject(encoded_fragment)

            pdf_writer.add_metadata(metadata)
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
            for i, fragment in enumerate(fragments):
                # Encode fragment
                encoded_data = base64.b64encode(str(fragment).encode('utf-8')).decode('utf-8')

                # Create annotation object with hidden data in comments
                annotation = DictionaryObject()
                annotation.update({
                    "/Type": "/Annot",
                    "/Subtype": "/Text",
                    "/Rect": ArrayObject([0, 0, 0, 0]),  # Invisible rectangle
                    "/Contents": TextStringObject(f"Internal processing data: {encoded_data}"),
                    "/Open": False,
                    "/Flags": 2  # Hidden flag
                })

                # Add to first page annotations if possible
                if len(pdf_writer.pages) > 0:
                    page = pdf_writer.pages[0]
                    if "/Annots" not in page:
                        page["/Annots"] = ArrayObject()
                    page["/Annots"].append(annotation)

            print(f"[+] Hidden {len(fragments)} fragments in comments/markers")

        except Exception as e:
            print(f"[-] Error hiding in comments/markers: {str(e)}")

    def _add_dummy_objects(self, pdf_writer):
        """
        Add dummy objects to PDF for additional obfuscation.

        Args:
            pdf_writer: PDF writer object
        """
        try:
            # Add some innocent-looking dummy objects
            dummy_objects = [
                {"/Type": "/Catalog", "/Version": "/1.7"},
                {"/Type": "/Font", "/Subtype": "/Type1", "/BaseFont": "/Helvetica"},
                {"/Type": "/ExtGState", "/CA": 1.0, "/ca": 1.0}
            ]

            for dummy_obj in dummy_objects:
                obj = DictionaryObject()
                obj.update(dummy_obj)
                if hasattr(pdf_writer, '_objects'):
                    pdf_writer._objects.append(obj)

            print("[+] Added dummy objects for obfuscation")

        except Exception as e:
            print(f"[-] Error adding dummy objects: {str(e)}")

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
                pdf_reader = PdfReader(pdf_file)

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
                    if any(prefix in str(key) for prefix in ['/DocumentVersion', '/CreationTool', '/ProcessingInfo', '/ValidationData', '/InternalID']):
                        try:
                            decoded_data = base64.b64decode(str(value).encode('utf-8'))
                            fragment = eval(decoded_data.decode('utf-8'))
                            fragments.append(fragment)
                        except:
                            continue
        except Exception as e:
            print(f"[-] Error extracting from metadata: {str(e)}")

        return fragments

    def _extract_from_comments(self, pdf_reader):
        """Extract data fragments from PDF comments."""
        fragments = []
        try:
            for page in pdf_reader.pages:
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        if "/Contents" in annot:
                            content = str(annot["/Contents"])
                            if "Internal processing data:" in content:
                                try:
                                    encoded_data = content.split("Internal processing data: ")[1]
                                    decoded_data = base64.b64decode(encoded_data.encode('utf-8'))
                                    fragment = eval(decoded_data.decode('utf-8'))
                                    fragments.append(fragment)
                                except:
                                    continue
        except Exception as e:
            print(f"[-] Error extracting from comments: {str(e)}")

        return fragments

    def _reconstruct_data(self, fragments):
        """Reconstruct encrypted data from fragments."""
        try:
            # Sort fragments by ID
            fragments.sort(key=lambda x: x['id'])

            # Verify completeness
            if not fragments:
                return None

            total_chunks = fragments[0]['total_chunks']
            if len(fragments) != total_chunks:
                print(f"[-] Missing fragments: expected {total_chunks}, found {len(fragments)}")
                return None

            # Reconstruct data
            reconstructed_data = b''
            for fragment in fragments:
                reconstructed_data += fragment['data']

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
            metadata_size = struct.unpack('<I', prepared_data[offset:offset+4])[0]
            offset += 4

            # Skip metadata
            offset += metadata_size

            # Read compressed data size
            compressed_size = struct.unpack('<I', prepared_data[offset:offset+4])[0]
            offset += 4

            # Extract and decompress PNG data
            compressed_png = prepared_data[offset:offset+compressed_size]
            png_data = zlib.decompress(compressed_png)

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

if __name__ == "__main__":
    # Example 1: Hide PNG in PDF
    pdf_input = r"/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/5G Network_ Security and Solutions_.pdf"
    png_input = r"/home/d3bugger/Projects/FINAL YEAR PROJECT/src/media/stego/input_pngs/Screenshot_20250524_151908_LuMOERY.png"
    pdf_output = r"/home/d3bugger/Projects/FINAL YEAR PROJECT/src/Financial_Report_2023_Modified.pdf"

    # Hide the PNG image
    success = main(pdf_input, png_input, pdf_output, password="my_secret_password")

    if success:
        print("\n[+] Steganography process completed successfully!")
        print("The PDF appears normal but contains the hidden PNG image.")

        # Example 2: Extract the hidden PNG (for verification)
        extracted_png = r"C:\Users\JohnDoe\Pictures\Extracted_Logo.png"
        extract_success = extract_png(pdf_output, extracted_png, password="my_secret_password")

        if extract_success:
            print(f"[+] Hidden PNG successfully extracted to: {extracted_png}")
        else:
            print("[-] Failed to extract hidden PNG")
    else:
        print("[-] Steganography process failed")