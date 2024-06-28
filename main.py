import streamlit as st
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import numpy as np
import sqlite3
import cv2
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None  
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
conn = sqlite3.connect(':memory:')
c = conn.cursor()
c.execute('''
    CREATE TABLE datapoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unique_id TEXT,
        filing_number TEXT,
        filing_date TEXT,
        rcs_number TEXT,
        dp_value TEXT,
        dp_unique_value TEXT
    )
''')
def fix_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
def zoom_image(image, zoom_factor=1.5):
    height, width = image.shape[:2]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return zoomed
def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text
def resize_image(image, max_width=1000):
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_dimensions = (max_width, int(height * ratio))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return image
def detect_checkboxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.8 < aspect_ratio < 1.2 and 10 < w < 40 and 10 < h < 40:
            roi = image[y:y+h, x:x+w]
            filled = cv2.countNonZero(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) > (0.5 * w * h)
            checkboxes.append((x, y, w, h, filled))
    
    return checkboxes
def draw_checkboxes(image, checkboxes):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h, filled) in checkboxes:
        color = "green" if filled else "red"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    return image
def process_pdf(file):
    images = convert_from_path(file)
    extracted_data = []

    for page_num, image in enumerate(images):
        image = np.array(image)
        image = fix_rotation(image)
        image = enhance_image(image)
        image = zoom_image(image)
        image = resize_image(image)

        text = extract_text_from_image(image)
        checkboxes = detect_checkboxes(image)
        image_with_boxes = draw_checkboxes(Image.fromarray(image), checkboxes)
        filing_type, dp_sections = identify_filing_type_and_sections(text, checkboxes)

        extracted_data.append({
            "page": page_num + 1,
            "text": text,
            "checkboxes": checkboxes,
            "image": image_with_boxes,
            "filing_type": filing_type,
            "dp_sections": dp_sections
        })

    return extracted_data
def identify_filing_type_and_sections(text, checkboxes):
    filing_type = "Unknown"
    dp_sections = []

    if any(checkbox[4] for checkbox in checkboxes): 
        filing_type = "Specific Filing Type"
        dp_sections = ["Section 1", "Section 2"] 

    return filing_type, dp_sections
def main():
    st.title("PDF Data Extraction Engine Demo")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            extracted_data = process_pdf("temp.pdf")
            st.subheader("Extracted Data Points:")
            for data in extracted_data:
                st.write(f"**Page {data['page']}**")
                st.text(data['text'])

                st.subheader("Detected Checkboxes")
                for x, y, w, h, filled in data['checkboxes']:
                    st.write(f"Checkbox at ({x}, {y}), Size ({w}x{h}), Filled: {'Yes' if filled else 'No'}")

                st.subheader("Filing Type and Sections")
                st.write(f"Filing Type: {data['filing_type']}")
                st.write(f"Detected Sections: {', '.join(data['dp_sections'])}")

                st.image(data['image'], caption=f"Page {data['page']} with Checkboxes")

                c.execute('''
                    INSERT INTO datapoints (unique_id, filing_number, filing_date, rcs_number, dp_value, dp_unique_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    str(data['page']),
                    "Filing Number Placeholder",  
                    "Filing Date Placeholder",
                    "RCS Number Placeholder",
                    data['text'],
                    "DP Unique Value Placeholder"
                ))

            conn.commit()
            st.subheader("Review and Adjust Data Points")
            datapoints = pd.read_sql_query("SELECT * FROM datapoints", conn)
            edited_df = st.experimental_data_editor(datapoints, num_rows="dynamic")
            if st.button("Save Adjustments"):
                edited_df.to_sql('datapoints', conn, if_exists='replace', index=False)
                st.success("Adjustments saved successfully.")
            csv = datapoints.to_csv(index=False).encode('utf-8')
            st.download_button("Download Extracted Data as CSV", data=csv, file_name="extracted_data.csv")

if __name__ == "__main__":
    main()
