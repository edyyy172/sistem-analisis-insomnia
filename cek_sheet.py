import pandas as pd

df = pd.read_excel(
    "dataset sleep health.xlsx",
    sheet_name="Sheet3"
)

print(df[[
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level"
]].head(10))