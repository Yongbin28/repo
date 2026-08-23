# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('streamlit_app.py', '.'), ('ms-playwright', 'ms-playwright'), ('TesterFamilyMap.xlsx', '.'), ('icon.jpg', 'icon.jpg')]
if os.path.exists("cookies.pkl"):
    datas.append(("cookies.pkl", "."))
# If user has an ico file, should include it too. 
# Looking at code, it expects app_icon.ico
if os.path.exists("app_icon.ico"):
    datas.append(("app_icon.ico", "."))
binaries = []
hiddenimports = ['ADYAP_DataPulling', 'MDM_WaferId', 'MDM_CFC_DLog_Extraction', 'ml_train_model', 'ml_compute_statistic', 'wafer_data_combiner', 'ml_yield_prediction', 'partname_mapping', 'save_cookies', 'stdf_decryptor', 'utils', 'app_summary', 'plotly', 'joblib', 'openpyxl', 'lightgbm', 'scipy', 'reliability_grading', 'spatial_analysis', 'runtime_calibration', 'selenium.webdriver.chrome.options', 'selenium.webdriver.chrome.service', 'selenium.webdriver.common.by', 'selenium.webdriver.support.ui', 'selenium.webdriver.support.expected_conditions']
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('playwright')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('xgboost')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('catboost')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('selenium')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Wafer_Quality_Gate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Wafer_Quality_Gate',
)
