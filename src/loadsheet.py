import pandas as pd
import sqlalchemy

# =========================================================
# 1. MYSQL CONNECTION
# =========================================================
username = "root"
password = "your_new_password"
host = "localhost"
database = "poc"

engine = sqlalchemy.create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)

# =========================================================
# 2. LOAD EXCEL FILE
# =========================================================
excel_path = "/home/wizzgeeks/Downloads/APEXECISE_SAMPLE_DATA_13102025_UPDATED_V2.0.xlsx"
xlsx = pd.ExcelFile(excel_path)

# =========================================================
# 3. LOOP THROUGH SHEETS AND WRITE TO TABLES
# =========================================================
for sheet_name in xlsx.sheet_names:
    print(f"Loading sheet: {sheet_name}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Replace spaces, uppercase → lowercase
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )

    df.to_sql(
        name=sheet_name.lower(),
        con=engine,
        if_exists="replace",   # change to append if needed
        index=False,
        chunksize=5000
    )

    print(f"✔ Loaded: {sheet_name} → table `{sheet_name.lower()}`")

print("🎉 All sheets loaded into MySQL successfully!")
