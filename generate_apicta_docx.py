import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Set standard margins (1 inch)
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Set base style to Calibri
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)
font.color.rgb = RGBColor(0x22, 0x22, 0x22)

# Colors
PRIMARY = RGBColor(0x02, 0x24, 0x31)  # Deep Navy
SECONDARY = RGBColor(0x00, 0x66, 0x99)
GRAY = RGBColor(0x55, 0x55, 0x55)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

# Title Header
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_title = p_title.add_run('MD (Malaysia Digital) APICTA Awards 2026\nNomination Submission Form')
run_title.font.name = 'Calibri'
run_title.font.size = Pt(18)
run_title.font.bold = True
run_title.font.color.rgb = PRIMARY

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_sub = p_sub.add_run('Category: Tertiary Student Project | Individual Submission')
run_sub.font.name = 'Calibri'
run_sub.font.size = Pt(12)
run_sub.font.bold = True
run_sub.font.color.rgb = SECONDARY

p_meta = doc.add_paragraph()
p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_meta = p_meta.add_run('Nominee: Lek Yong Bin (TP071541) • Asia Pacific University of Technology & Innovation (APU)')
run_meta.font.name = 'Calibri'
run_meta.font.size = Pt(10.5)
run_meta.font.italic = True
run_meta.font.color.rgb = GRAY

doc.add_paragraph() # Spacer

# Content Data - Fully Synchronized with Architecture Document
sections = [
    ('1. School / University Name *', 'Asia Pacific University of Technology & Innovation (APU)'),
    ('2. Billing Address *', 'Asia Pacific University of Technology & Innovation (APU),\nTechnology Park Malaysia (TPM), Bukit Jalil,\n57000 Kuala Lumpur, Federal Territory of Kuala Lumpur, Malaysia.'),
    ('3. Solution / Service / Project Name *', 'WaferPulse: Explainable AI and Statistical Quality Gating for Early Semiconductor Reliability Risk Interception'),
    ('4. Contact Person Name (1) *', 'Lek Yong Bin'),
    ('5. Mobile Contact *', '+6011-55010028'),
    ('6. Email *', 'TP071541@mail.apu.edu.my'),
    ('7. Contact Person (2)', 'N/A (Individual Submission)'),
    ('8. Email (2)', 'N/A'),
    ('9. Mobile Contact (2)', 'N/A'),
    ('10. Please select the Category you are submitting *', 'Tertiary Student Project'),
    ('11. Brief Profile of Student / School / University (max 200 words) *',
     'Student: Lek Yong Bin is a final-year Bachelor of Engineering (BEng Hons) in Electrical and Electronic Engineering undergraduate at Asia Pacific University of Technology & Innovation (APU), Malaysia. Specializing in semiconductor automated test engineering, data analytics, and applied artificial intelligence, he conducts applied research focused on extracting predictive intelligence from complex Automated Test Equipment (ATE) datasets.\n\n'
     'School & University: Asia Pacific University of Technology & Innovation (APU) is a premier digital technology university in Malaysia, globally accredited under the Board of Engineers Malaysia (BEM) and the Washington Accord. APU\'s School of Engineering prioritizes real-world industrial relevance, complex engineering problem-solving (Washington Accord WP1–WP7), and AI integration into high-value manufacturing.\n\n'
     'Inspired by the challenge of latent post-probe failures in advanced semiconductor packaging, Lek Yong Bin independently conceptualized, engineered, and validated the WaferPulse platform under university academic guidance. The project demonstrates how an individual Malaysian undergraduate engineer can develop robust, enterprise-grade AI and statistical pipelines to uncover hidden reliability risks at the wafer probing stage. (166 words)'),
    ('12. Synopsis of Solution / Services / Project *',
     'WaferPulse is an explainable AI and intelligent quality-gating platform designed to uncover latent semiconductor reliability risks at the wafer probe testing stage. In high-volume manufacturing, marginal dies that pass standard probe test specifications can harbor subtle parametric anomalies that remain completely hidden during wafer sort, only to fail later when subjected to severe thermomechanical packaging stress and Joule-heating-induced electromigration. In advanced 3D IC and multi-die stacking, assembling a latently compromised die destroys the entire multi-chip package. WaferPulse bridges this critical testing gap by analyzing raw wafer probe test data (STDF, STDF.gz, XFS, DLOG) through partial wafer retest combination, 19-feature statistical extraction, dual-layered drift detection (SPC & PSI), spatial Good-Die–Bad-Neighbour (GDBN) screening, multi-algorithm ML benchmarking, and TreeSHAP explainability. By predicting downstream final test performance and reliability scores before packaging, WaferPulse intercepts hidden risks at the probe stage, ensuring true Known Good Die (KGD) integrity.'),
    ('13. Description of Solution / Services / Project *',
     'A major vulnerability in semiconductor manufacturing is the hidden defect gap that exists between wafer probe testing and downstream assembly. Conventional automated test equipment (ATE) applies rigid, single-parameter upper and lower limits. Marginal dies that technically pass these static limits are passed as \"good.\" However, latent defects remain hidden during probing and only manifest downstream under packaging thermomechanical stresses, leading to localized Joule heating, electromigration, and package failure. In advanced 3D multi-die integration, discovering a defective die after packaging is economically catastrophic, as the entire integrated assembly must be scrapped.\n\n'
     'Guided by Lean Six Sigma and DMADV (Define, Measure, Analyze, Design, Verify) engineering principles to eliminate Cost of Poor Quality (COPQ), WaferPulse solves this challenge through an end-to-end software pipeline:\n\n'
     '1. Tester Data Decryption & Partial Wafer Combination: Directly parses binary test data across standard formats (STDF, STDF.gz, Teradyne XFS, DLOG) and intelligently aggregates interrupted or multi-pass retest logs into unified die coordinate records.\n\n'
     '2. 19-Parameter Feature Engineering: Extracts 19 mathematical descriptors across 4 groups—Basic Statistics (mean, std, median, IQR, min, max), Data Quality (missing rate, outlier rate at median ± 3×IQR), Quantile Distributions (p01 to p99), and Limit Proximity (PPM near LSL/USL)—compressing millions of data points into high-signal wafer vectors.\n\n'
     '3. Dual-Layered Dynamic Shift Detection: Implements dynamic Part Average Testing (PAT) using Statistical Process Control (SPC) for central tendency and Population Stability Index (PSI) to detect subtle lot-level distribution shifts against historical golden baselines.\n\n'
     '4. Spatial Defect Screening (GDBN): Employs Good-Die–Bad-Neighbour algorithms to flag electrically functional dies surrounded by failing neighbors that carry high latent failure probabilities.\n\n'
     '5. Predictive ML & Explainability: Trains and benchmarks 9 algorithms (Ridge, Lasso, ElasticNet, KNN, RandomForest, ExtraTrees, XGBoost, HistGBR, LightGBM) under cross-stage correlation, automatically deploying the top 3 models to predict Final Test statistics and yield, with TreeSHAP explainers identifying the exact shifted parameters.'),
    ('14. Functionalities and Features or Execution, Team Composition & Implementation or Proof of how solution will Scale *',
     'A. Key Functionalities & Features:\n'
     '• Multi-Format ATE Compatibility: Native decryption of STDF, STDF.gz, Teradyne XFS, and DLOG formats without vendor-locked third-party licenses.\n'
     '• Automated 9-Algorithm Benchmarking: Concurrently evaluates 9 diverse ML regressors with train/validation splitting, ranking models via MAE, RMSE, and R² to automatically deploy the top 3 performers.\n'
     '• Dynamic Quality & Drift Gating: Dual-layer SPC and PSI distribution shift detection against historical golden baselines.\n'
     '• Spatial Defect Interception: Wafer-map GDBN filtering to eliminate high-risk dies adjacent to defect clusters.\n'
     '• 4-Tier Predictive Product Grading: Mathematically classifies incoming wafers based on reliability risk scores:\n'
     '  - Grade A: Automotive (Score ≥ 90, ultra-tight zero-defect criteria)\n'
     '  - Grade B: Industrial (80 ≤ Score < 90, robust industrial tolerance)\n'
     '  - Grade C: Consumer (60 ≤ Score < 80, wider consumer tolerance)\n'
     '  - Grade D: Reject (Score < 60, excluded before packaging)\n'
     '• Explainable AI (XAI): TreeSHAP attribution plots revealing root-cause parameters driving anomaly flags.\n\n'
     'B. Implementation & Execution:\n'
     '• Conceived, architected, and coded independently by the student using Python, NumPy, Pandas, Scikit-learn, LightGBM, SHAP, and Streamlit.\n'
     '• Implemented as a modular monolith separating processing, intelligence, and presentation layers, backed by 33 automated regression test cases.\n\n'
     'C. Proof of Scalability:\n'
     '• High Computational Feasibility: Feature engineering compresses raw die logs into 19 statistical metrics, drastically reducing data footprint and eliminating the need for expensive GPU clusters.\n'
     '• Sub-Second Batch Throughput: Ingests 25-wafer lot cassettes and executes full statistical gating and ML predictions in seconds on standard factory PCs.\n'
     '• Industrial COPQ Reduction: Eliminates geometric scrap in advanced 3D packaging by intercepting latent risks at the earliest possible stage (\"Shift-Left\" methodology).'),
    ('15. Local Contents Statement (To confirm major design & development are done locally by the local entity) *',
     'I hereby confirm and declare that 100% of the conceptual design, architectural engineering, mathematical formulation, algorithm development, software coding, and system validation of WaferPulse were independently performed locally in Malaysia by Lek Yong Bin, an undergraduate engineering student at Asia Pacific University of Technology & Innovation (APU), Technology Park Malaysia, Bukit Jalil, Kuala Lumpur. No foreign commercial software cores or outsourced third-party proprietary systems were utilized in the creation of this solution.'),
    ('16. Attached Document Link (if any)',
     'https://drive.google.com/drive/folders/1iIlFVlQPAO9pBP4ZXFRtPkYHTkYZkeIQ?usp=sharing'),
    ('17. Payment Information (Reference)',
     'Beneficiary: PIKOM\nBank Name: Malayan Banking Bhd\nBank A/C No: 514150327167\nBank Address: Wisma PKNS, Jalan Raja Laut Tambahan, 50350 KL\nBank SWIFT Code: MBBEMYKL\nEntry Fee: RM100 (Tertiary Academia)'),
    ('18. Declaration *',
     '☑ We declare that we own the Intellectual Property Rights (IPR) of this product.')
]

for q, a in sections:
    table = doc.add_table(rows=2, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell_q = table.cell(0, 0)
    cell_q.width = Inches(6.5)
    set_cell_background(cell_q, 'EBF3F9')
    set_cell_margins(cell_q, top=100, bottom=100, left=150, right=150)
    p_q = cell_q.paragraphs[0]
    p_q.paragraph_format.space_before = Pt(2)
    p_q.paragraph_format.space_after = Pt(2)
    run_q = p_q.add_run(q)
    run_q.font.name = 'Calibri'
    run_q.font.size = Pt(11)
    run_q.font.bold = True
    run_q.font.color.rgb = PRIMARY

    cell_a = table.cell(1, 0)
    cell_a.width = Inches(6.5)
    set_cell_background(cell_a, 'FFFFFF')
    set_cell_margins(cell_a, top=120, bottom=120, left=150, right=150)
    p_a = cell_a.paragraphs[0]
    p_a.paragraph_format.space_before = Pt(2)
    p_a.paragraph_format.space_after = Pt(4)
    p_a.paragraph_format.line_spacing = 1.15

    paras = a.split('\n\n')
    for idx, para in enumerate(paras):
        if idx > 0:
            p_a = cell_a.add_paragraph()
            p_a.paragraph_format.space_before = Pt(2)
            p_a.paragraph_format.space_after = Pt(4)
            p_a.paragraph_format.line_spacing = 1.15
        lines = para.split('\n')
        for l_idx, line in enumerate(lines):
            if l_idx > 0:
                p_a.add_run('\n')
            run_text = p_a.add_run(line)
            run_text.font.name = 'Calibri'
            run_text.font.size = Pt(10.5)
            run_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    doc.add_paragraph()

out_path = 'd:/APU/Micron/code -Micron(latest version)/APICTA_2026_Nomination_Form_Lek_Yong_Bin.docx'
doc.save(out_path)
print('Updated document saved successfully to:', out_path)
