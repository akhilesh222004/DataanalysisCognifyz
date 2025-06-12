import pandas as pd            #Task1&2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv ("C:\\Users\\HP\\OneDrive\\Desktop\\CognifyzTasks\\Data_set 2 - Copy.csv")
print(df)
print(df.info())
print(df.describe())
print(df['gender'].value_counts())

sns.set(style="whitegrid")  

plt.figure(figsize=(6, 4))  
sns.countplot(x='gender', data=df, palette='pastel')  

plt.title('Gender Distribution - Bar Chart')  
plt.xlabel('Gender')  
plt.ylabel('Count')   
plt.tight_layout()    
plt.show()            

gender_counts = df['gender'].value_counts()

plt.figure(figsize=(6, 6))  # Chart size
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
        colors=['#ff9999', '#66b3ff'])

plt.title('Gender Distribution - Pie Chart')
plt.tight_layout()
plt.show()
#Task3
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Columns:")
print(numerical_columns)
for col in numerical_columns:
    print(f"\nStatistics for column: {col}")
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Standard Deviation: {df[col].std():.2f}")

    #Task4
    print(df.columns)
investment_columns = ['Mutual_Funds', 'Equity_Market', 'Gold', 'Real_Estate', 'Fixed_Deposits', 'Government_Bonds','Stock_Marktet','PPF']
investment_columns = [col for col in investment_columns if col in df.columns]
investment_columns = [col for col in investment_columns if col in df.columns]
investment_counts = df[investment_columns].sum().sort_values(ascending=False)

print("Investment Preferences (Total Selections):")
print(investment_counts)
most_preferred = investment_counts.idxmax()
most_preferred_count = investment_counts.max()
print(f"\n Most Preferred Investment Avenue: {most_preferred} ({most_preferred_count} people)")


plt.figure(figsize=(8, 5))
sns.barplot(x=investment_counts.values, y=investment_counts.index, palette="viridis")
plt.title("Most Preferred Investment Avenues")
plt.xlabel("Number of People")
plt.ylabel("Investment Avenue")
plt.tight_layout()
plt.show()

#Task5
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
reason_columns = [col for col in df.columns if 'reason' in col.lower()]
combined_reasons = df[reason_columns].astype(str).apply(lambda row: ' '.join(row.values), axis=1)
reasons_text = ' '.join(combined_reasons)
reasons_text = re.sub(r'[^a-zA-Z\s]', '', reasons_text).lower()  # remove punctuation and lowercase
words = reasons_text.split()
filtered_words = [word for word in words if word not in stop_words]
reason_counts = Counter(filtered_words)
print("Top 10 Common Reasons for Investment Across All Avenues:")
for word, count in reason_counts.most_common(10):
    print(f"• {word}: {count}")
top_reasons_df = pd.DataFrame(reason_counts.most_common(10), columns=['Reason', 'Count'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Reason', data=top_reasons_df, palette='viridis')
plt.title('Top 10 Common Reasons for Investment')
plt.xlabel('Frequency')
plt.ylabel('Reason')
plt.tight_layout()
plt.show()
#Task6
custom_stopwords = {'plan', 'plans', 'saving', 'savings', 'objective', 'objectives', 'goal', 'goals'}
stop_words.update(custom_stopwords)

multi_word_phrases = {
    "health care": "health_care",
    "emergency fund": "emergency_fund",
    "child education": "child_education",
    "higher education": "higher_education",
    "retirement planning": "retirement_planning",
    "medical expenses": "medical_expenses",
    "home purchase": "home_purchase"
}


column_name = "What are your savings objectives?"

if column_name not in df.columns:
    raise KeyError(f"Column '{column_name}' not found in dataset.")


text = ' '.join(df[column_name].dropna().astype(str)).lower()


for phrase, replacement in multi_word_phrases.items():
    text = text.replace(phrase, replacement)


text = re.sub(r'[^a-zA-Z_\s]', '', text)


words = text.split()
filtered_words = [word for word in words if word not in stop_words]

objective_counts = Counter(filtered_words)

top_objectives = [(word.replace('_', ' ').capitalize(), count) for word, count in objective_counts.most_common(10)]

print("\nTop 10 Refined Savings Objectives (Grouped Phrases Fixed):")
for word, count in top_objectives:
    print(f"• {word}: {count}")

    top_objectives_df = pd.DataFrame(top_objectives, columns=['Objective', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Objective', data=top_objectives_df, palette='viridis')
plt.title('Top 10 Savings Objectives (with Grouped Phrases)')
plt.xlabel('Frequency')
plt.ylabel('Objective')
plt.tight_layout()
plt.show()
#Task7
custom_stopwords = {'information', 'sources', 'source', 'investment', 'investments', 'decision', 'decisions', 'regarding'}
stop_words.update(custom_stopwords)

info_col = [col for col in df.columns if "source" in col.lower() or "information" in col.lower()]
if not info_col:
    raise ValueError("No column found with 'source' or 'information' in column name.")
column_name = info_col[0]

info_text = ' '.join(df[column_name].dropna().astype(str)).lower()

phrases = {
    "financial consultant": "financial_consultant",
    "mutual fund agent": "mutual_fund_agent",
    "Television": "Television",
    "social media": "social_media",
    "news paper": "news_paper",
    "financial advisor": "financial_advisor"
}
for phrase, replacement in phrases.items():
    info_text = info_text.replace(phrase, replacement)

info_text = re.sub(r'[^a-zA-Z_\s]', '', info_text)

words = info_text.split()
filtered_words = [word for word in words if word not in stop_words]

source_counts = Counter(filtered_words)

top_sources = [(word.replace('_', ' ').capitalize(), count) for word, count in source_counts.most_common(10)]


print("\nTop 10 Common Sources of Investment Information (Grouped Correctly):")
for word, count in top_sources:
    print(f"• {word}: {count}")


top_sources_df = pd.DataFrame(top_sources, columns=['Source', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Source', data=top_sources_df, palette='magma')
plt.title('Top 10 Grouped Investment Information Sources')
plt.xlabel('Frequency')
plt.ylabel('Source')
plt.tight_layout()
plt.show()
#Task8
df['Duration_clean'] = df['Duration'].astype(str).apply(lambda x: re.findall(r'\d+', x))
df['Duration_clean'] = df['Duration_clean'].apply(lambda x: int(x[0]) if x else None)

valid_durations = df['Duration_clean'].dropna()
average_duration = valid_durations.mean()
print(f"Average Investment Duration: {average_duration:.2f} years")

plt.figure(figsize=(10, 6))
sns.histplot(valid_durations, bins=10, kde=True, color='skyblue')
plt.title('Distribution of Investment Durations')
plt.xlabel('Investment Duration (Years)')
plt.ylabel('Number of Participants')
plt.axvline(average_duration, color='red', linestyle='--', label=f'Average: {average_duration:.2f} yrs')
plt.legend()
plt.tight_layout()
plt.show()
#Task9
objectives = df['Objective'].dropna().astype(str)
full_text = ' '.join(objectives).lower()
full_text = re.sub(r'[^a-zA-Z\s]', '', full_text)  


phrases = [
    'capital appreciation',
    'tax saving',
    'wealth creation',
    'retirement planning',
    'income generation',
    'financial security',
    'growth',
    'safety',
    'returns',
    'children education',
    'emergency fund',
    'income'  
]


phrase_counts = {}
for phrase in phrases:
    count = full_text.count(phrase)
    if count > 0:
        phrase_counts[phrase.title()] = count  


print("Top Investment Expectations:")
for phrase, count in phrase_counts.items():
    print(f"• {phrase}: {count}")


plt.figure(figsize=(10, 6))
plt.bar(phrase_counts.keys(), phrase_counts.values(), color='darkcyan')
plt.title('Top Investment Expectations')
plt.xlabel('Expectation')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#Task10
df_corr = df.copy()


label_cols = ['gender', 'Duration', 'What are your savings objectives?', 'Objective',
              'Reason_Equity', 'Reason_Mutual', 'Reason_Bonds', 'Reason_FD', 'Avenue']


le = LabelEncoder()
for col in label_cols:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))


if 'Expected Return' in df_corr.columns:
    corr_columns = ['age', 'gender', 'Duration', 'What are your savings objectives?', 'Objective',
                    'Reason_Equity', 'Reason_Mutual', 'Reason_Bonds', 'Reason_FD',
                    'Avenue', 'Expected Return']
else:
    corr_columns = ['age', 'gender', 'Duration', 'What are your savings objectives?', 'Objective',
                    'Reason_Equity', 'Reason_Mutual', 'Reason_Bonds', 'Reason_FD', 'Avenue']


corr_matrix = df_corr[corr_columns].corr()


plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Investment Dataset')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

