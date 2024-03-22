import pandas as pd

# Path to your CSV file
csv_file_path = r"C:\Users\rmada\Downloads\AAPL_news.csv"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Drop rows where the content column is null
df = df.dropna(subset=['content'])

# Drop rows where the content column is an empty string
df = df[df['content'].str.strip() != '']

# List of strings to check for in the content column
unwanted_strings = ["Error", "Not Supported Yet", "Connection Error"]

# Remove rows containing any of the unwanted strings in the content column
for unwanted in unwanted_strings:
    df = df[~df['content'].str.contains(unwanted, case=False, na=False)]

# Resulting DataFrame after cleaning
print(df.head())

# If you need to save the cleaned DataFrame back to a CSV
df.to_csv(r'C:\Users\rmada\Downloads\AAPL_news_cleaned.csv', index=False)
