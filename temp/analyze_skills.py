import json, re, os

with open('data/reference/krm_disciplines_09.03.02.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
    data = raw['09.03.02']

# Find ML discipline
target = None
for name, disc in data['disciplines'].items():
    if 'Python' in name and 'машин' in name.lower():
        target = (name, disc)
        break

if not target:
    # find any with ПК-7
    for name, disc in data['disciplines'].items():
        if 'ПК-7' in disc['skills']:
            target = (name, disc)
            break

if not target:
    print('NOT FOUND')
    exit()

name, disc = target
print(f'DISCIPLINE: {name}')
print(f'Competencies: {list(disc["skills"].keys())}')
print()

for comp, skills in disc['skills'].items():
    print(f'\n=== {comp} ({len(skills)} skills) ===')
    for i, s in enumerate(skills):
        print(f'  [{i:2d}] {s}')

print('\n\n---')
print(f'Total disciplines: {len(data["disciplines"])}')
total_skills = sum(len(skills) for disc in data['disciplines'].values() for skills in disc['skills'].values())
print(f'Total skills: {total_skills}')
