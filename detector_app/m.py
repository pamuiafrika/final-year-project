import os
import sys
import time
from l import hide_png_in_pdf
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_pdf(args):
    input_pdf, output_pdf, png_path, password, technique = args
    success, out_path = hide_png_in_pdf(
        pdf_path=input_pdf,
        png_path=png_path,
        output_path=output_pdf,
        password=password,
        technique=technique
    )
    return (input_pdf, out_path, success)

def bulk_hide_png_in_pdfs(input_folder, output_folder, png_path, password=None, technique="metadata"):
    """
    Hide a PNG in all PDF files in input_folder and save to output_folder.
    Args:
        input_folder (str): Path to folder with input PDFs
        output_folder (str): Path to folder for output stego-PDFs
        png_path (str): Path to PNG to hide
        password (str): Optional password for encryption
        technique (str): Hiding technique
    Returns:
        List of (input_pdf, output_pdf, success) tuples
    """
    os.makedirs(output_folder, exist_ok=True)
    tasks = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            input_pdf = os.path.join(input_folder, filename)
            output_pdf = os.path.join(output_folder, filename.replace('.pdf', '_stego.pdf'))
            tasks.append((input_pdf, output_pdf, png_path, password, technique))
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pdf, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk PDF Steganography: Hide PNG in all PDFs in a folder.")
    parser.add_argument('--input_folder', required=True, help='Folder with input PDFs')
    parser.add_argument('--output_folder', required=True, help='Folder to save stego-PDFs')
    parser.add_argument('--png', required=True, help='PNG image to hide')
    parser.add_argument('--password', default=None, help='Password for encryption (optional)')
    parser.add_argument('--technique', default='metadata', help='Hiding technique (metadata, comments, objects, whitespace, multi)')
    args = parser.parse_args()

    # Check if input and output directories exist
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist.")
        sys.exit(1)

    start_time = time.time()
    print(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    results = bulk_hide_png_in_pdfs(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        png_path=args.png,
        password=args.password,
        technique=args.technique
    )
    print("\nBulk processing complete. Results:")
    for inp, outp, ok in results:
        print(f"{inp} -> {outp} : {'SUCCESS' if ok else 'FAILED'}")

    # Generate and print a summary report
    end_time = time.time()
    elapsed = end_time - start_time
    total = len(results)
    successes = sum(1 for _, _, ok in results if ok)
    failures = total - successes
    failed_files = [inp for inp, _, ok in results if not ok]
    print("\n=== Bulk Processing Report ===")
    print(f"Total PDFs processed: {total}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    if failures > 0:
        print("Failed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

