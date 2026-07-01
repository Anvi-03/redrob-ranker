import pandas as pd

df = pd.read_csv('../data/labeled_candidates.csv')

def auto_label(row):
    title = str(row['current_title']).lower()
    summary = str(row['summary']).lower()
    exp = row['years_of_experience']
    
    # 0 = Not a fit (HR, Marketing, Civil, Content Writer, QA, etc.)
    bad_roles = ['hr', 'marketing', 'civil', 'content', 'accountant', 'qa', 'graphic']
    if any(b in title for b in bad_roles):
        return 0
        
    # 3 = Perfect fit (ML/AI Engineer, 5-9 years experience)
    good_roles = ['machine learning', 'ml', 'ai', 'data scientist', 'artificial intelligence']
    is_ml = any(g in title for g in good_roles) or any(g in summary for g in good_roles)
    
    if is_ml and 5 <= exp <= 9:
        return 3
    elif is_ml and (3 <= exp < 5 or exp > 9):
        return 2
    elif 'software engineer' in title or 'backend' in title:
        return 1
    else:
        return 0

df['label'] = df.apply(auto_label, axis=1)
df.to_csv('../data/labeled_candidates.csv', index=False)
print("Auto-labeled 200 candidates successfully!")
