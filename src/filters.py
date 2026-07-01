import re

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'deloitte', 'pwc', 'kpmg', 'ey', 'hcl', 'tech mahindra'
}

def is_honeypot(cand):
    """
    Detects honeypot profiles based on logically impossible data.
    """
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])

    # Honeypot 1: Expert proficiency with 0 duration used
    expert_zero_duration_count = 0
    for s in skills:
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
            expert_zero_duration_count += 1
    if expert_zero_duration_count >= 3:
        return True

    # Honeypot 2: A single job duration is longer than their total years of experience
    total_yoe_months = profile.get('years_of_experience', 0) * 12
    for job in career:
        duration = job.get('duration_months', 0)
        if duration > total_yoe_months + 12: # Add 1 year buffer for rounding
            return True

    return False

def is_disqualified(cand):
    """
    Rule-based disqualifications based on JD explicit criteria.
    """
    career = cand.get('career_history', [])
    if not career:
        return True # Cannot evaluate without career history

    # Check 1: Purely consulting firms
    # If EVERY job they've ever had is at a known pure consulting firm
    all_consulting = True
    for job in career:
        comp = job.get('company', '').lower()
        if not any(consulting_firm in comp for consulting_firm in CONSULTING_FIRMS):
            all_consulting = False
            break
    if all_consulting:
        return True

    # Check 2: Purely academic/research environments
    # If all titles are purely researcher, postdoc, graduate student, etc. AND no product company
    academic_titles = {'researcher', 'postdoc', 'phd', 'graduate research', 'research assistant', 'fellow'}
    all_academic = True
    for job in career:
        title = job.get('title', '').lower()
        industry = job.get('industry', '').lower()
        if not any(ac in title for ac in academic_titles) and 'academic' not in industry:
            all_academic = False
            break
    if all_academic:
        return True

    return False
