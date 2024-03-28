import pandas as pd

def replace_characters(column_value):
    # ทำการแทนที่ตัวอักษรตามที่ระบุด้วยเลข
    replaced_string = column_value.replace('n', '1').replace('c', '2').replace('g', '3').replace('r', '4').replace('p', '5').replace('u', '6').replace('e', '8').replace('w', '9').replace('y', '10').replace('p', '5')
    
    
    # แปลงเป็นตัวเลข
    numeric_result = int(replaced_string)
    
    return numeric_result

# อ่านข้อมูลจาก Excel
file_path = 'clean/mushroom cleanning.xlsx'  # แทนที่ด้วยที่อยู่ของไฟล์ Excel ของคุณ
df = pd.read_excel(file_path)

# ใช้ apply() เพื่อทำการแทนที่ตัวอักษรในหลักที่ 2
df[1] = df[1].apply(replace_characters)

# บันทึก DataFrame เป็นไฟล์ Excel ใหม่
output_file_path = 'convert\test.xlsx'  # กำหนดที่อยู่ของไฟล์ Excel ที่จะบันทึก
df.to_excel(output_file_path, index=False)
