from cx_Freeze import setup, Executable

build_options = {
    "packages": [
        "streamlit",
        "webview",
        "sklearn",
        "xgboost",
        "catboost",
        "lightgbm",
        "plotly",
        "playwright",
        "joblib",
        "openpyxl",
        "scipy"
    ],
    "include_files": [
        ("ms-playwright", "ms-playwright"),
        ("streamlit_app.py", "streamlit_app.py"),
        ("TesterFamilyMap.xlsx", "TesterFamilyMap.xlsx")
    ]
}

setup(
    name="Wafer Quality Gate Latest",
    version="1.0",
    description="Wafer Quality Gate Desktop App",
    options={"build_exe": build_options},
    executables=[Executable("desktop_app.py", base="gui")]
)
