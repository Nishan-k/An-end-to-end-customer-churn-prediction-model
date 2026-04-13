import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import re

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re

import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

def markdown_to_reportlab(text):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=16, alignment=TA_CENTER, spaceAfter=12)
    heading1_style = ParagraphStyle('Heading1', parent=styles['Heading1'], fontSize=14, spaceAfter=6)
    heading2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, spaceAfter=6)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, spaceAfter=4)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=10, leftIndent=20, bulletIndent=10)

    def convert_inline(md):
        # Replace **bold** with <b>bold</b>
        md = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', md)
        # Replace *italic* with <i>italic</i>
        md = re.sub(r'\*(.*?)\*', r'<i>\1</i>', md)
        return md

    flowables = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 6))
            continue

        # Headings
        if line.startswith('# '):
            content = convert_inline(line[2:])
            flowables.append(Paragraph(content, title_style))
        elif line.startswith('## '):
            content = convert_inline(line[3:])
            flowables.append(Paragraph(content, heading1_style))
        elif line.startswith('### '):
            content = convert_inline(line[4:])
            flowables.append(Paragraph(content, heading2_style))
        # Bullet points (both - and *)
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = convert_inline(line[2:])
            # If bullet already has a number or dash, keep as is
            flowables.append(Paragraph(f'• {bullet_text}', bullet_style))
        # Numbered list (e.g., "1. ...")
        elif re.match(r'^\d+\.', line):
            # Convert numbered line to bullet style (or keep as normal)
            content = convert_inline(line)
            flowables.append(Paragraph(content, normal_style))
        else:
            # Normal paragraph
            content = convert_inline(line)
            flowables.append(Paragraph(content, normal_style))
    return flowables

def save_report_as_pdf(report_text: str, pdf_filename: str) -> str:
    """Save report as PDF using ReportLab."""
    reports_dir = "generated_reports"
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    story = markdown_to_reportlab(report_text)
    doc.build(story)
    return pdf_path