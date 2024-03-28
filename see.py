import pandas as pd

# อ่านข้อมูลจาก Excel
data = pd.read_excel('convert/mushroom cleanning tranform to number2.csv')

data_cleaned = data.dropna()

# บันทึก DataFrame ที่ไม่มีค่า NaN หรือค่าว่างเป็นไฟล์ CSV
data_cleaned.to_csv('ข้อมูล_cleaned.csv', index=False)
