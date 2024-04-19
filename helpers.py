import subprocess
import os
import glob

def open_pdf(file_path):
    try:
        # Open the PDF file in the default PDF viewer
        subprocess.run(['start', '', file_path], check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unable to open PDF file: {e}")

def run_pdflatex(file_name):
    try:
        subprocess.run(['pdflatex', file_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unable to run pdflatex: {e}")

def cleanup_files(path):
    patterns = ['*.aux', '*.log', '*.vscodeLog', '*.tex']
    for pattern in patterns:
        files_to_delete = glob.glob(os.path.join(path, pattern))
        for file in files_to_delete:
            os.remove(file)
