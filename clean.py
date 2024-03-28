import pandas as pd


excel_path = 'data/test mu.xlsx'
df = pd.read_excel(excel_path)


df = df[~df.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1)]


output_excel_path = 'clean/test mu cl.xlsx'
df.to_excel(output_excel_path, index=False)

print(f"Rows with '?' removed. Output saved to {output_excel_path}")