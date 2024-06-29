import fitz  
import pytesseract
from PIL import Image
import cv2
import numpy as np
import sqlite3
import streamlit as st
import os  
def init_db():
    conn = sqlite3.connect('extracted_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY,
            filing_number TEXT,
            filing_date TEXT,
            rcs_number TEXT,
            dp_key TEXT,
            dp_value TEXT,
            dp_unique_value TEXT
        )
    ''')
    conn.commit()
    conn.close()
def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned PDF using OCR."""
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        page_text = pytesseract.image_to_string(gray)
        text += page_text + "\n"
    
    pdf_document.close()
    return text
def detect_checkboxes(image):
    """Detect checkboxes in an image using OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    checkboxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:  
                checkboxes.append((x, y, w, h))
    return checkboxes
def extract_and_process_checkboxes(pdf_path):
    """Extract and process checkboxes from a scanned PDF."""
    pdf_document = fitz.open(pdf_path)
    checkboxes_all_pages = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_np = np.array(img)

        checkboxes = detect_checkboxes(image_np)
        checkboxes_all_pages.append(checkboxes)
    
    pdf_document.close()
    return checkboxes_all_pages
def extract_dps_from_text(text, anchors):
    """Extract data points based on the text and given anchors."""
    extracted_dps = {}
    for anchor, dp_position in anchors.items():
        anchor_index = text.find(anchor)
        if anchor_index != -1:
            dp_start = anchor_index + len(anchor) + dp_position['offset']
            dp_end = dp_start + dp_position['length']
            extracted_dps[anchor] = text[dp_start:dp_end].strip()
    return extracted_dps
anchors = {
    "Invoice Number:": {"offset": 1, "length": 20},  
    "Date:": {"offset": 1, "length": 10}
}
def save_extracted_data(filing_number, filing_date, rcs_number, dps):
    conn = sqlite3.connect('extracted_data.db')
    cursor = conn.cursor()
    for dp_key, dp_value in dps.items():
        cursor.execute('''
            INSERT INTO extracted_data (filing_number, filing_date, rcs_number, dp_key, dp_value, dp_unique_value)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filing_number, filing_date, rcs_number, dp_key, dp_value, f"{dp_key}_{dp_value[:10]}"))
    conn.commit()
    conn.close()
def get_data_by_filing_number(filing_number):
    conn = sqlite3.connect('extracted_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM extracted_data WHERE filing_number = ?', (filing_number,))
    data = cursor.fetchall()
    conn.close()
    return data

init_db()
st.title("PDF Data Extraction Engine")
uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    pdf_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("Extracting text from the uploaded PDF...")
    pdf_text = extract_text_from_scanned_pdf(pdf_path)
    st.text_area("Extracted Text", pdf_text, height=200)
    st.write("Detecting checkboxes in the PDF...")
    checkboxes = extract_and_process_checkboxes(pdf_path)
    st.write("Checkboxes detected (coordinates on each page):", checkboxes)
    st.write("Extracting data points based on anchors...")
    dps = extract_dps_from_text(pdf_text, anchors)
    st.json(dps)
    filing_number = st.text_input("Filing Number")
    filing_date = st.text_input("Filing Date")
    rcs_number = st.text_input("RCS Number")

    if st.button("Save Data"):
        save_extracted_data(filing_number, filing_date, rcs_number, dps)
        st.success("Data saved successfully!")
st.header("Retrieve Extracted Data")
filing_number_query = st.text_input("Enter Filing Number to Query")
if st.button("Get Data"):
    queried_data = get_data_by_filing_number(filing_number_query)
    if queried_data:
        st.write("Extracted Data for Filing Number:", filing_number_query)
        st.write(queried_data)
    else:
        st.write("No data found for the given filing number.")
