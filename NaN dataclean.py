import pandas as pd

# อ่านข้อมูลจาก Excel
data = pd.read_excel('convert/mushroom cleanning tranform to number2.xlsx')

# ลบแถวที่มีค่า NaN
data_cleaned = data.dropna()

# พิมพ์ DataFrame หลังจากลบค่า NaN
print(data_cleaned)

# บันทึก DataFrame ที่ไม่มีค่า NaN เป็นไฟล์ Excel
data_cleaned.to_excel('DataNumber_cleaned.xlsx', index=False)