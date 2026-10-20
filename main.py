
from pathlib import Path
import pandas as pd

# 1. Load the dataset
data_path = Path("data/enron_spam_data.csv")
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# 2. Check the target labels
print("\nOriginal label distribution:")
print(df["Spam/Ham"].value_counts(dropna=False))

# 3. Handle missing subjects and messages
df["Subject"] = df["Subject"].fillna("")
df["Message"] = df["Message"].fillna("")

# 4. Combine subject and body into one text feature
df["Email_Text"] = (
    df["Subject"].astype(str).str.strip()
    + " "
    + df["Message"].astype(str).str.strip()
).str.strip()

# 5. Remove emails with no usable text
df = df[df["Email_Text"].str.len() > 0].copy()

# 6. Normalize labels and remove unexpected labels
df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.strip().str.lower()
df = df[df["Spam/Ham"].isin(["spam", "ham"])].copy()

# 7. Convert labels to numbers
df["Label"] = df["Spam/Ham"].map({"ham": 0, "spam": 1})

# 8. Display the cleaned dataset
print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df[["Email_Text", "Label"]].isnull().sum())

print("\nCleaned label distribution:")
print(df["Spam/Ham"].value_counts())

print("\nSample cleaned emails:")
print(df[["Email_Text", "Spam/Ham", "Label"]].head(3).to_string(index=False))

print("\nData cleaning completed successfully!")




from sklearn.model_selection import train_test_split

# Input features and target labels
X = df["Email_Text"]
y = df["Label"]

# Split: 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN/TEST SPLIT ---")
print("Total emails:", len(X))
print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))
print("Training label distribution:")
print(y_train.value_counts().rename(index={0: "Ham", 1: "Spam"}))
print("\nData split completed successfully!")
