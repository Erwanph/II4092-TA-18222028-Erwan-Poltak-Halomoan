import calendar
from pathlib import Path

import pandas as pd

m2_dir = Path(__file__).parent.parent / "raw" / "bps" / "m2"
m2_data = {}
months = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
    'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
    'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}
month_indices = list(range(1, 13))

for year in range(2010, 2026):
    file_path = m2_dir / f"Uang Beredar, {year}.csv"
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        continue
    print(f"Processing {file_path.name}...")
    
    try:
        df = pd.read_csv(file_path, header=None)
        m2_row_idx = None
        for idx, row in df.iterrows():
            if pd.notna(row.iloc[0]) and 'M2' in str(row.iloc[0]):
                m2_row_idx = idx
                break
        if m2_row_idx is None:
            print(f"  Error: Could not find M2 row in {file_path.name}")
            continue
        
        m2_row = df.iloc[m2_row_idx]
        
        # Extract M2 values for each month
        for month_name, month_num in months.items():
            col_idx = month_num
            if col_idx < len(m2_row):
                m2_value = m2_row.iloc[col_idx]
                if pd.isna(m2_value) or m2_value == '-' or m2_value == '':
                    continue
                if month_num == 2:
                    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                    day = 29 if is_leap else 28
                else:
                    day = calendar.monthrange(year, month_num)[1]
                
                period = f"{year:04d}-{month_num:02d}-{day:02d}"
                try:
                    m2_value_float = float(m2_value) / 1000
                    m2_data[period] = m2_value_float
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")

# Load FRED data for 2005-2009 and earlier months to fill gaps
print("\nLoading FRED data from MYAGM2IDM189N.csv...")
fred_file = Path(__file__).parent.parent / "processed" / "MYAGM2IDM189N.csv"

if fred_file.exists():
    try:
        fred_df = pd.read_csv(fred_file)
        for _, row in fred_df.iterrows():
            period_str = str(row['period'])
            m2_rp = row['M2_Rp']
            if pd.isna(m2_rp):
                continue
            date_obj = pd.to_datetime(period_str)
            year = date_obj.year
            month = date_obj.month

            if month == 2:
                is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                day = 29 if is_leap else 28
            else:
                day = calendar.monthrange(year, month)[1]

            period = f"{year:04d}-{month:02d}-{day:02d}"
            
            # Convert from Rupiah to Triliun Rp (divide by 1 trillion = 1e12)
            try:
                m2_value_float = float(m2_rp) / 1e12
                if period not in m2_data:
                    m2_data[period] = m2_value_float
            except (ValueError, TypeError):
                pass
        
        print(f"  Added FRED data: {len([p for p in m2_data.keys() if p < '2010-01-01'])} periods before 2010")
    except Exception as e:
        print(f"  Error loading FRED data: {e}")
else:
    print(f"  Warning: FRED file not found at {fred_file}")

output_data = []

start_year, start_month = 2005, 7
end_year, end_month = 2025, 12

current_year = start_year
current_month = start_month

while (current_year, current_month) <= (end_year, end_month):
    if current_month == 2:
        is_leap = (current_year % 4 == 0 and current_year % 100 != 0) or (current_year % 400 == 0)
        day = 29 if is_leap else 28
    else:
        day = calendar.monthrange(current_year, current_month)[1]
    
    period = f"{current_year:04d}-{current_month:02d}-{day:02d}"
    
    m2_value = m2_data.get(period, "")
    
    output_data.append({
        'period': period,
        'M2_Triliun_Rp': m2_value
    })
    
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

output_df = pd.DataFrame(output_data)

output_file = Path(__file__).parent.parent / "raw" / "external" / "M2_Triliun_Rp.csv"
output_df.to_csv(output_file, index=False)

print(f"\nOutput saved to: {output_file}")
print(f"Total records: {len(output_df)}")
print(f"Records with M2 values: {len(output_df[output_df['M2_Triliun_Rp'] != ''])}")
print("\nFirst few rows:")
print(output_df.head(10))
print("\nLast few rows:")
print(output_df.tail(10))
