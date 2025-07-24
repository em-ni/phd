import os
from pdf2image import convert_from_path

def convert_pdfs_to_pngs(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(dirpath, filename)
                try:
                    images = convert_from_path(pdf_path, dpi=600)
                    for i, image in enumerate(images):
                        png_filename = f"{os.path.splitext(filename)[0]}_converted.png"
                        png_path = os.path.join(dirpath, png_filename)
                        image.save(png_path, 'PNG')
                    print(f"Converted: {pdf_path}")
                except Exception as e:
                    print(f"Error converting {pdf_path}: {e}")

if __name__ == "__main__":
    input_folder = r"C:\Users\z5440219\OneDrive - UNSW\Desktop\github\phd\slam\rpg_trajectory_evaluation\em\broncho\phantom"  # Change this to your folder path
    convert_pdfs_to_pngs(input_folder)