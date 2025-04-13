import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Electric_Vehicle_Population_Uncleaned.csv")


df['Make'] = df['Make'].str.strip().str.upper()
df['Model'] = df['Model'].str.strip().str.title()
df['City'] = df['City'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.strip()
df['Postal Code'] = df['Postal Code'].astype(str).str.extract(r'(\d{5})')
df['Model Year'] = df['Model Year'].astype(str).str.extract(r'(\d{4})').astype(float)

df.drop_duplicates(inplace=True)
df.dropna(subset=['County', 'City', 'Electric Vehicle Type', 'Electric Utility'], inplace=True)

df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')


sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))


plt.subplot(2, 2, 1)
top_makes = df['Make'].value_counts().head(10)
top_makes_df = top_makes.reset_index()
top_makes_df.columns = ['Make', 'Count']
top_makes_df['Hue'] = top_makes_df['Make']  
sns.barplot(data=top_makes_df, x='Count', y='Make', hue='Hue', palette="viridis", legend=False)
plt.title("Top 10 EV Makes")


plt.subplot(2, 2, 2)
ev_type_counts = df['Electric Vehicle Type'].value_counts()
plt.pie(ev_type_counts.values, labels=ev_type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Electric Vehicle Type Distribution")

plt.subplot(2, 2, 3)
range_by_year = df.groupby('Model Year')['Electric Range'].mean().dropna()
sns.lineplot(x=range_by_year.index, y=range_by_year.values, marker='o')
plt.title("Avg Electric Range by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Average Range (mi)")

plt.subplot(2, 2, 4)
top_counties = df['County'].value_counts().head(10)
top_counties_df = top_counties.reset_index()
top_counties_df.columns = ['County', 'Count']
top_counties_df['Hue'] = top_counties_df['County']  
sns.barplot(data=top_counties_df, x='Count', y='County', hue='Hue', palette="coolwarm", legend=False)
plt.title("Top 10 Counties by EV Registrations")

plt.figure(figsize=(8, 6))
avg_price_by_type = df.groupby('Electric Vehicle Type')['Base MSRP'].mean().dropna().sort_values(ascending=False)
avg_price_df = avg_price_by_type.reset_index()
avg_price_df['Hue'] = avg_price_df['Electric Vehicle Type']

sns.barplot(data=avg_price_df, x='Base MSRP', y='Electric Vehicle Type', hue='Hue', palette='Set2', legend=False)
plt.title("Average Base MSRP by EV Type")
plt.xlabel("Average MSRP ($)")
plt.ylabel("EV Type")

plt.figure(figsize=(14, 6))

sns.boxplot(data=df, x='Model Year', y='Electric Range', color='lightblue')

plt.figure(figsize=(14, 6))
sns.barplot(data=df, x='Model', y='Electric Range')
plt.title('Electric Range Distribution by Model')
plt.xlabel('Model')
plt.ylabel('Electric Range (miles)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

yearly_stats = df.groupby('Model Year')[['Electric Range', 'Base MSRP']].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_stats, x='Model Year', y='Electric Range', label='Avg Electric Range', marker='o')
sns.lineplot(data=yearly_stats, x='Model Year', y='Base MSRP', label='Avg Base MSRP', marker='s')
plt.title('Trends in Electric Range and Base MSRP Over Time')
plt.xlabel('Model Year')
plt.ylabel('Average Value')
plt.grid(True)

plt.tight_layout()
plt.show()


