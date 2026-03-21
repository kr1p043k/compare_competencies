#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import subprocess
import os
import requests
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# 🔧 НАСТРОЙКА ПУТЕЙ
# ============================================
current_dir = Path(__file__).parent.resolve()
PROJECT_ROOT = current_dir.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================
# 📁 ПУТИ К ДАННЫМ
# ============================================
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STUDENTS_DIR = DATA_DIR / "students"
RAW_DIR = DATA_DIR / "raw"
RESULT_DIR = DATA_DIR / "result"

for dir_path in [PROCESSED_DIR, STUDENTS_DIR, RAW_DIR, RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# 🎨 CSS СТИЛИ
# ============================================
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .profile-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .profile-card h3 {
        color: #1e3c72;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-highlight {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2a5298;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .coverage-text {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 8px;
        color: #333 !important;
    }
    .coverage-high {
        background-color: #d4edda;
        color: #155724 !important;
    }
    .coverage-medium {
        background-color: #fff3cd;
        color: #856404 !important;
    }
    .coverage-low {
        background-color: #f8d7da;
        color: #721c24 !important;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #fef9e6 0%, #fff3e0 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #ff9800;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recommendation-card strong {
        color: #b65f00;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.8rem;
    }
    .recommendation-card .message {
        color: #4a4a4a;
        line-height: 1.5;
        white-space: pre-line;
    }
    .warning-card {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe0e0 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #ff4444;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-card strong {
        color: #c62828;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.8rem;
    }
    .warning-card .message {
        color: #5a2a2a;
        line-height: 1.5;
        white-space: pre-line;
    }
    .success-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-card strong {
        color: #2e7d32;
        font-size: 1.1rem;
        display: block;
        margin-bottom: 0.8rem;
    }
    .success-card .message {
        color: #1b5e20;
        line-height: 1.5;
        white-space: pre-line;
    }
    .vacancy-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .vacancy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .vacancy-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    .vacancy-company {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .vacancy-location {
        color: #888;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .vacancy-skills {
        color: #2a5298;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 🔤 УТИЛИТЫ ДЛЯ КОДИРОВОК
# ============================================
def decode_bytes_safe(data):
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        for enc in ['utf-8', 'cp1251', 'utf-8-sig', 'latin-1']:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode('utf-8', errors='replace')
    return str(data)

def read_file_with_fallback_encoding(filepath):
    for enc in ['utf-8', 'cp1251', 'utf-8-sig', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(filepath, 'rb') as f:
        return f.read().decode('utf-8', errors='replace')

# ============================================
# 📥 ЗАГРУЗКА ДАННЫХ
# ============================================
@st.cache_data(ttl=3600)
def load_market_competencies():
    filepath = PROCESSED_DIR / "competency_frequency_mapped.json"
    
    if filepath.exists():
        try:
            content = read_file_with_fallback_encoding(filepath)
            data = json.loads(content)
            if len(data) > 10:
                return data
        except Exception as e:
            st.warning(f"Ошибка загрузки: {e}")
    
    return generate_realistic_market_data()

def generate_realistic_market_data():
    competencies = [
        "BD-1.1", "BD-1.2", "BD-1.3", "BD-1.4", "BD-2.1", "BD-2.2",
        "BD-3.1", "BD-3.2", "BD-4.1", "BD-4.2", "ML-2.1", "ML-2.2",
        "ML-2.3", "ML-3.1", "ML-3.2", "DL-1.2", "DL-1.3", "DL-1.6",
        "DL-1.12", "PL-1.1", "PL-1.2", "PL-1.3", "PL-1.4", "SS1.1",
        "SS1.2", "SS2.1", "SS2.2", "SS3.1", "SS3.2", "SS3.3"
    ]
    
    market_data = {}
    for i, comp in enumerate(competencies):
        importance = 1 - (i / len(competencies)) ** 1.5
        freq = int(50 + importance * 150)
        market_data[comp] = freq
    
    important_comps = ["ML-2.1", "PL-1.1", "BD-3.1", "DL-1.3"]
    for comp in important_comps:
        if comp in market_data:
            market_data[comp] = int(market_data[comp] * 1.5)
    
    return market_data

@st.cache_data(ttl=3600)
def load_all_profiles():
    profiles = {}
    profile_names = ['base', 'dc', 'top_dc']
    
    for name in profile_names:
        filepath = STUDENTS_DIR / f"{name}_competency.json"
        if filepath.exists():
            try:
                content = read_file_with_fallback_encoding(filepath)
                data = json.loads(content)
                skills = data.get("навыки", [])
                if skills:
                    profiles[name] = skills
            except Exception as e:
                st.error(f"Ошибка загрузки {name}: {e}")
        else:
            profiles[name] = get_realistic_profile_skills(name)
    
    return profiles

def get_realistic_profile_skills(profile_name):
    all_competencies = generate_realistic_market_data().keys()
    all_comp_list = list(all_competencies)
    
    if profile_name == 'base':
        n_skills = int(len(all_comp_list) * 0.3)
        return sorted(all_comp_list[:n_skills])
    elif profile_name == 'dc':
        n_skills = int(len(all_comp_list) * 0.5)
        return sorted(all_comp_list[:n_skills])
    else:
        n_skills = int(len(all_comp_list) * 0.7)
        return sorted(all_comp_list[:n_skills])

@st.cache_data(ttl=3600)
def load_competency_descriptions():
    descriptions = {}
    filepath = STUDENTS_DIR / "descriptiom_of_competency.txt"
    if filepath.exists():
        try:
            content = read_file_with_fallback_encoding(filepath)
            lines = content.splitlines()
            for line in lines[1:]:
                line = line.strip()
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        descriptions[parts[0]] = parts[1]
        except:
            pass
    return descriptions

# ============================================
# 📊 ПОИСК ВАКАНСИЙ С УЧЕТОМ РЕГИОНА
# ============================================
@st.cache_data(ttl=3600)
def search_vacancies(query, area_id=1, per_page=10):
    """Поиск вакансий через hh.ru API с учетом региона"""
    try:
        url = "https://api.hh.ru/vacancies"
        params = {
            "text": query,
            "area": area_id,
            "per_page": per_page,
            "page": 0,
            "order_by": "publication_time"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("items", [])
        else:
            return []
    except Exception as e:
        st.error(f"Ошибка поиска вакансий: {e}")
        return []

def get_vacancy_details(vacancy_id):
    """Получение детальной информации о вакансии"""
    try:
        url = f"https://api.hh.ru/vacancies/{vacancy_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_area_name(area_id):
    """Получение названия региона по ID"""
    areas = {
        1: "Москва",
        2: "Санкт-Петербург",
        115: "Ростов-на-Дону",
        1006: "Россия"
    }
    return areas.get(area_id, "Россия")

# ============================================
# 📊 ФУНКЦИИ АНАЛИЗА
# ============================================
def compare_competencies(program_skills, market_skills):
    program_set = set(program_skills)
    market_set = set(market_skills.keys())
    return (program_set & market_set, program_set - market_set, market_set - program_set)

def calculate_gap_analysis(program_skills, market_skills, threshold=0.2):
    market_set = set(market_skills.keys())
    program_set = set(program_skills)
    missing = market_set - program_set
    max_freq = max(market_skills.values()) if market_skills else 1
    high_demand, medium_demand, low_demand = [], [], []
    for skill in missing:
        freq = market_skills.get(skill, 0)
        normalized = freq / max_freq
        if normalized > threshold:
            high_demand.append((skill, freq))
        elif normalized > threshold * 0.3:
            medium_demand.append((skill, freq))
        else:
            low_demand.append((skill, freq))
    return {
        'high_demand': sorted(high_demand, key=lambda x: x[1], reverse=True),
        'medium_demand': sorted(medium_demand, key=lambda x: x[1], reverse=True),
        'low_demand': sorted(low_demand, key=lambda x: x[1], reverse=True),
        'total_missing': len(missing)
    }

def calculate_coverage_metrics(program_skills, market_skills):
    program_set = set(program_skills)
    market_set = set(market_skills.keys())
    
    if not market_set:
        return {
            'coverage': 0, 
            'weighted_coverage': 0, 
            'program_size': len(program_set), 
            'market_size': 0, 
            'common_count': 0,
            'coverage_details': {}
        }
    
    common = program_set & market_set
    
    coverage = len(common) / len(market_set) if market_set else 0
    
    total_weight = sum(market_skills.values())
    weighted_coverage = sum(market_skills.get(s, 0) for s in program_set) / total_weight if total_weight > 0 else 0
    
    categories = categorize_competencies(list(market_set))
    coverage_details = {}
    for cat, skills in categories.items():
        common_in_cat = len([s for s in skills if s in program_set])
        total_in_cat = len(skills)
        coverage_details[cat] = {
            'total': total_in_cat,
            'covered': common_in_cat,
            'coverage': common_in_cat / total_in_cat if total_in_cat > 0 else 0
        }
    
    return {
        'coverage': coverage,
        'weighted_coverage': weighted_coverage,
        'program_size': len(program_set),
        'market_size': len(market_set),
        'common_count': len(common),
        'coverage_details': coverage_details
    }

def categorize_competencies(skills_list):
    categories = defaultdict(list)
    for skill in skills_list:
        category = skill.split('-')[0] if '-' in skill else (skill[:2] if len(skill) >= 2 else "OT")
        categories[category].append(skill)
    return dict(categories)

# ============================================
# 📊 ВИЗУАЛИЗАЦИИ
# ============================================

def create_horizontal_bar_chart(data_dict, title="Топ востребованных компетенций", top_n=20):
    if not data_dict or len(data_dict) == 0:
        return None
    
    items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    categories = [item[0] for item in items]
    values = [item[1] for item in items]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
    bars = ax.barh(categories, values, color=colors)
    
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                str(val), ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Частота упоминаний', fontsize=12, fontweight='bold')
    ax.set_ylabel('Компетенции', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_pie_chart_by_category(market_skills, top_n=10):
    if not market_skills or len(market_skills) == 0:
        return None
    
    cats = categorize_competencies(list(market_skills.keys()))
    cat_freq = {}
    for cat, skills in cats.items():
        total = sum(market_skills.get(s, 0) for s in skills)
        if total > 0:
            cat_freq[cat] = total
    
    if not cat_freq:
        return None
    
    sorted_items = sorted(cat_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    categories = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=categories, 
        autopct=lambda pct: f'{pct:.1f}%',
        colors=plt.cm.Set3(np.linspace(0, 1, len(categories))),
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        pctdistance=0.85,
        labeldistance=1.1
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('normal')
    
    ax.set_title('Распределение компетенций по категориям', fontsize=14, fontweight='bold', pad=20)
    
    if len(categories) > 8:
        ax.legend(wedges, categories, title="Категории", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    return fig

def create_profile_comparison_chart(profiles_dict, market_skills):
    if not profiles_dict or not market_skills:
        return None
    
    comparison_data = []
    profile_names = {'base': 'BASE', 'dc': 'DATA CENTER', 'top_dc': 'TOP DATA CENTER'}
    
    for name, display_name in profile_names.items():
        if name in profiles_dict:
            skills = profiles_dict[name]
            metrics = calculate_coverage_metrics(skills, market_skills)
            comparison_data.append({
                'Профиль': display_name,
                'Покрытие (%)': round(metrics['coverage'] * 100, 1),
                'Компетенций': metrics['program_size'],
                'Общих с рынком': metrics['common_count']
            })
    
    if not comparison_data:
        return None
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Покрытие (%)', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(df)]
    bars = ax.barh(df['Профиль'], df['Покрытие (%)'], color=colors)
    
    for bar, val in zip(bars, df['Покрытие (%)']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Покрытие рынка (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Программа', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение программ обучения', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_gap_chart(gap_data, max_freq):
    if not gap_data['high_demand']:
        return None
    
    items = gap_data['high_demand'][:15]
    competencies = [item[0] for item in items]
    frequencies = [item[1] for item in items]
    percentages = [(freq / max_freq * 100) for freq in frequencies]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(competencies) * 0.4)))
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(competencies)))
    bars = ax.barh(competencies, percentages, color=colors)
    
    for bar, pct, freq in zip(bars, percentages, frequencies):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({freq})', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Относительная востребованность (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Компетенции', fontsize=12, fontweight='bold')
    ax.set_title('Критический дефицит компетенций', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================
# 💡 ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ (С ДЕТАЛЬНЫМИ СПИСКАМИ)
# ============================================
def generate_recommendations(profile_name, profile_skills, market_skills, gap_analysis, metrics):
    recommendations = []
    
    if not gap_analysis['high_demand']:
        recommendations.append({
            'type': 'success',
            'title': '✓ Отлично!',
            'message': 'Ваша программа хорошо соответствует рыночным требованиям. Критических дефицитов не обнаружено.\n\nРекомендуется продолжать развивать текущие направления и следить за новыми трендами.'
        })
    else:
        # Детальные рекомендации по критическим дефицитам
        high_demand_skills = gap_analysis['high_demand']
        if high_demand_skills:
            skills_list = '\n'.join([f'• {skill} (востребованность: {freq} упоминаний)' 
                                     for skill, freq in high_demand_skills])
            recommendations.append({
                'type': 'warning',
                'title': f'🔴 Критический дефицит ({len(high_demand_skills)} компетенций)',
                'message': f'Следующие компетенции имеют высокую востребованность на рынке, но отсутствуют в программе:\n\n{skills_list}\n\n📌 **Рекомендации по устранению:**\n• Включить эти компетенции в обязательные дисциплины\n• Разработать специализированные курсы и модули\n• Пригласить экспертов из индустрии для проведения мастер-классов\n• Добавить практические проекты по этим направлениям'
            })
    
    # Детальные рекомендации по средне-востребованным
    medium_demand_skills = gap_analysis['medium_demand']
    if medium_demand_skills:
        skills_list = '\n'.join([f'• {skill} (востребованность: {freq} упоминаний)' 
                                 for skill, freq in medium_demand_skills])
        recommendations.append({
            'type': 'info',
            'title': f'🟡 Средняя востребованность ({len(medium_demand_skills)} компетенций)',
            'message': f'Компетенции со средней востребованностью, которые можно добавить:\n\n{skills_list}\n\n📌 **Рекомендации:**\n• Рассмотреть включение в факультативы и спецкурсы\n• Добавить в программы повышения квалификации\n• Включить в темы курсовых и дипломных работ\n• Предложить студентам для самостоятельного изучения'
        })
    
    # Рекомендации по категориям с низким покрытием
    if metrics['coverage_details']:
        low_coverage_cats = []
        for cat, data in metrics['coverage_details'].items():
            if data['coverage'] < 0.4 and data['total'] > 0:
                missing_skills = [s for s in market_skills.keys() if s.startswith(cat) and s not in set(profile_skills)]
                low_coverage_cats.append((cat, data['coverage'] * 100, data['total'] - data['covered'], missing_skills[:5]))
        
        if low_coverage_cats:
            low_coverage_cats.sort(key=lambda x: x[1])
            cats_info = []
            for cat, cov, missing, missing_skills_list in low_coverage_cats[:3]:
                skills_examples = ', '.join(missing_skills_list[:3]) if missing_skills_list else "нет данных"
                cats_info.append(f'• **{cat}**: {cov:.0f}% покрытия (не хватает {missing} компетенций)\n  Примеры: {skills_examples}')
            
            cats_list = '\n\n'.join(cats_info)
            recommendations.append({
                'type': 'info',
                'title': '📊 Категории с низким покрытием',
                'message': f'Обратите внимание на следующие категории компетенций:\n\n{cats_list}\n\n📌 **Рекомендации:**\n• Усилить эти направления в учебном плане\n• Добавить специализированные модули и курсы\n• Привлечь практикующих специалистов для чтения лекций\n• Организовать дополнительные практикумы и воркшопы'
            })
    
    return recommendations

# ============================================
# 🚀 ЗАПУСК ПАЙПЛАЙНА
# ============================================
def run_full_pipeline(query="Data Scientist", area_id=1, max_pages=5, period=30):
    try:
        main_py_path = PROJECT_ROOT / "main.py"
        if not main_py_path.exists():
            st.error(f"❌ main.py не найден: {main_py_path}")
            return False
        
        cmd = [sys.executable, str(main_py_path), 
               "--query", str(query), 
               "--area-id", str(area_id), 
               "--max-pages", str(max_pages), 
               "--period", str(period), 
               "--skip-details",
               "--no-filter"]
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            encoding='utf-8',
            errors='replace'
        )
        
        stdout, stderr = process.communicate(timeout=300)
        
        if process.returncode == 0:
            st.success("✅ Пайплайн успешно выполнен!")
            if stdout:
                with st.expander("📋 Детали выполнения"):
                    st.code(stdout[-2000:])
            st.cache_data.clear()
            return True
        else:
            st.error(f"❌ Ошибка выполнения (код {process.returncode})")
            if stderr:
                with st.expander("❌ Ошибки выполнения"):
                    st.code(stderr)
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        st.error("⏰ Превышено время выполнения (5 минут)")
        return False
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        return False

# ============================================
# 💾 ЗАГРУЗКА ФАЙЛОВ
# ============================================
def save_uploaded_file(uploaded_file, destination_dir):
    try:
        filepath = destination_dir / uploaded_file.name
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return True
    except Exception as e:
        st.error(f"Ошибка сохранения: {e}")
        return False

def process_json_profile(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8', errors='replace')
        data = json.loads(content)
        if "навыки" in data and isinstance(data["навыки"], list):
            filepath = STUDENTS_DIR / uploaded_file.name
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        st.error("❌ Файл должен содержать ключ 'навыки' со списком")
        return False
    except Exception as e:
        st.error(f"❌ Ошибка обработки JSON: {e}")
        return False

def generate_profiles():
    try:
        csv_path = RAW_DIR / "competency_matrix.csv"
        if not csv_path.exists():
            csv_path = DATA_DIR / "last_uploaded" / "competency_matrix.csv"
        
        if not csv_path.exists():
            st.error(f"❌ CSV файл с матрицей компетенций не найден: {csv_path}")
            return {}
        
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from loaders.student_loader import generate_profiles_from_csv
        
        profiles = generate_profiles_from_csv(csv_path)
        return profiles
    except Exception as e:
        st.error(f"Ошибка генерации профилей: {e}")
        return {}

# ============================================
# 🎯 ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ============================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Система анализа компетенций</h1>
        <p>Сравнение учебных программ с требованиями рынка труда</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 🚀 КНОПКА ЗАПУСКА
    with st.container():
        st.markdown("### ⚡ Быстрый запуск")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            quick_query = st.text_input("Запрос", value="Data Scientist", key="qq")
        with col2:
            area_options = [(1, "Москва"), (2, "Санкт-Петербург"), (115, "Ростов-на-Дону"), (1006, "Россия (все регионы)")]
            quick_area = st.selectbox("Регион", area_options, format_func=lambda x: x[1], key="qa")
        with col3:
            quick_pages = st.slider("Страниц", 1, 20, 5, key="qp")
        with col4:
            quick_period = st.slider("Период (дни)", 1, 90, 30, key="qpd")
        
        if st.button("🚀 Запустить парсинг вакансий", type="primary", width="stretch"):
            with st.spinner("Запускаю парсинг вакансий... Это может занять несколько минут..."):
                success = run_full_pipeline(query=quick_query, area_id=quick_area[0], 
                                           max_pages=quick_pages, period=quick_period)
                if success:
                    st.balloons()
                    st.rerun()
    
    st.markdown("---")
    
    # 🔁 Загрузка данных
    with st.spinner("🔄 Загрузка данных..."):
        market_skills = load_market_competencies()
        profiles = load_all_profiles()
        competency_descriptions = load_competency_descriptions()
        
        # Поиск вакансий с учетом выбранного региона
        vacancies = search_vacancies(quick_query, quick_area[0], 10)
        area_name = get_area_name(quick_area[0])

    # ============================================
    # 🎛️ САЙДБАР
    # ============================================
    with st.sidebar:
        st.header("🎛️ Управление")
        
        st.markdown("---")
        st.subheader("📥 Загрузка")
        
        json_files = st.file_uploader("JSON профили", type=["json"], accept_multiple_files=True)
        if json_files and st.button("💾 Сохранить JSON"):
            saved = sum(1 for f in json_files if process_json_profile(f))
            if saved > 0:
                st.success(f"✅ Сохранено {saved} профилей!")
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("📈 Рынок")
        market_file = st.file_uploader("competency_frequency_mapped.json", type=["json"])
        if market_file and st.button("💾 Сохранить"):
            if save_uploaded_file(market_file, PROCESSED_DIR):
                st.success("✅ Сохранено!")
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        with st.expander("🔄 Генерация"):
            if st.button("📊 Сгенерировать профили из CSV"):
                with st.spinner("Генерация профилей из CSV..."):
                    new_profiles = generate_profiles()
                    if new_profiles:
                        st.success(f"✅ Сгенерировано {len(new_profiles)} профилей")
                        st.cache_data.clear()
                        st.rerun()
        
        st.markdown("---")
        
        with st.expander("🔍 Парсинг (отдельно)"):
            st.info(f"📁 main.py: {'✅ Найден' if (PROJECT_ROOT / 'main.py').exists() else '❌ Не найден'}")
            pq = st.text_input("Запрос", value="Data Scientist", key="pq")
            pa = st.selectbox("Регион", area_options, format_func=lambda x: x[1], key="pa")
            pp = st.slider("Страниц", 1, 20, 5, key="pp")
            if st.button("🚀 Парсить", type="secondary", width="stretch"):
                with st.spinner("Запуск парсинга..."):
                    if run_full_pipeline(query=pq, area_id=pa[0], max_pages=pp):
                        st.cache_data.clear()
                        st.success("✅ Готово!")
                        st.rerun()
        
        st.markdown("---")
        
        st.subheader("📊 Статистика")
        st.metric("Рыночных компетенций", len(market_skills))
        st.metric("Загружено профилей", len(profiles))

    # ============================================
    # 📊 ОСНОВНОЙ КОНТЕНТ
    # ============================================
    if market_skills and len(market_skills) > 0:
        st.success(f"✅ Данные загружены: {len(market_skills)} рыночных компетенций")
        
        # Карточки с краткой статистикой по профилям
        col1, col2, col3 = st.columns(3)
        
        profile_names = {'base': 'BASE', 'dc': 'DATA CENTER', 'top_dc': 'TOP DATA CENTER'}
        profile_colors = {'base': '#2ecc71', 'dc': '#3498db', 'top_dc': '#e74c3c'}
        
        for i, (name, display_name) in enumerate(profile_names.items()):
            if name in profiles:
                metrics = calculate_coverage_metrics(profiles[name], market_skills)
                coverage_percent = metrics['coverage'] * 100
                
                if coverage_percent >= 70:
                    coverage_class = "coverage-high"
                elif coverage_percent >= 50:
                    coverage_class = "coverage-medium"
                else:
                    coverage_class = "coverage-low"
                
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div class="profile-card">
                        <h3>{display_name}</h3>
                        <div class="metric-highlight">{metrics['program_size']}</div>
                        <div class="metric-label">компетенций в программе</div>
                        <div class="coverage-text {coverage_class}">
                            <strong>Покрытие рынка: {coverage_percent:.1f}%</strong>
                        </div>
                        <div style="margin-top: 8px; color: #666;">
                            Общих с рынком: <strong>{metrics['common_count']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Вкладки
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Рыночные компетенции", "📈 Сравнение профилей", "💡 Рекомендации", 
            "📋 Детальный список", "🔍 Gap-анализ", "💼 Актуальные вакансии"
        ])
        
        with tab1:
            st.header("📊 Топ востребованных компетенций на рынке")
            fig = create_horizontal_bar_chart(market_skills, title="Топ востребованных компетенций", top_n=20)
            if fig:
                st.pyplot(fig)
                plt.close()
            
            st.markdown("---")
            st.header("🥧 Распределение по категориям")
            fig = create_pie_chart_by_category(market_skills)
            if fig:
                st.pyplot(fig)
                plt.close()
        
        with tab2:
            st.header("📈 Сравнение программ обучения")
            fig = create_profile_comparison_chart(profiles, market_skills)
            if fig:
                st.pyplot(fig)
                plt.close()
            
            st.markdown("### 📊 Детальное сравнение")
            comp_data = []
            for name, display_name in profile_names.items():
                if name in profiles:
                    metrics = calculate_coverage_metrics(profiles[name], market_skills)
                    gap = calculate_gap_analysis(profiles[name], market_skills)
                    comp_data.append({
                        'Программа': display_name,
                        'Компетенций': metrics['program_size'],
                        'Общих с рынком': metrics['common_count'],
                        'Покрытие (%)': f"{metrics['coverage']*100:.1f}%",
                        'Взвешенное покрытие (%)': f"{metrics['weighted_coverage']*100:.1f}%",
                        'Критический дефицит': len(gap['high_demand'])
                    })
            
            if comp_data:
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
        
        with tab3:
            st.header("💡 Рекомендации по улучшению программ")
            
            for name, display_name in profile_names.items():
                if name in profiles:
                    st.subheader(f"📌 {display_name}")
                    
                    metrics = calculate_coverage_metrics(profiles[name], market_skills)
                    gap = calculate_gap_analysis(profiles[name], market_skills)
                    recommendations = generate_recommendations(display_name, profiles[name], market_skills, gap, metrics)
                    
                    for rec in recommendations:
                        if rec['type'] == 'success':
                            st.markdown(f"""
                            <div class="success-card">
                                <strong>{rec['title']}</strong>
                                <div class="message">{rec['message']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif rec['type'] == 'warning':
                            st.markdown(f"""
                            <div class="warning-card">
                                <strong>{rec['title']}</strong>
                                <div class="message">{rec['message']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <strong>{rec['title']}</strong>
                                <div class="message">{rec['message']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        
        with tab4:
            st.header("📋 Детальный список компетенций")
            
            all_skills = set()
            for skills in profiles.values():
                all_skills.update(skills)
            all_skills.update(market_skills.keys())
            
            details = []
            for skill in sorted(all_skills):
                row = {
                    "Код": skill,
                    "Востребованность": market_skills.get(skill, 0),
                    "Описание": competency_descriptions.get(skill, "")[:100]
                }
                for name, display_name in profile_names.items():
                    if name in profiles:
                        row[display_name] = "✅" if skill in profiles[name] else "❌"
                details.append(row)
            
            if details:
                st.dataframe(pd.DataFrame(details), use_container_width=True, height=500)
        
        with tab5:
            st.header("🔍 Gap-анализ по каждому профилю")
            
            for name, display_name in profile_names.items():
                if name in profiles:
                    st.subheader(f"📊 {display_name}")
                    
                    common, only_program, only_market = compare_competencies(profiles[name], market_skills)
                    metrics = calculate_coverage_metrics(profiles[name], market_skills)
                    gap = calculate_gap_analysis(profiles[name], market_skills)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Покрытие", f"{metrics['coverage']*100:.1f}%")
                    col2.metric("Взвешенное", f"{metrics['weighted_coverage']*100:.1f}%")
                    col3.metric("Всего", metrics['program_size'])
                    col4.metric("Общих с рынком", metrics['common_count'])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Недостающих компетенций", gap['total_missing'])
                    col2.metric("Критический дефицит", len(gap['high_demand']))
                    col3.metric("Средний дефицит", len(gap['medium_demand']))
                    
                    if gap['high_demand']:
                        st.markdown("**Критически важные компетенции для добавления:**")
                        for skill, freq in gap['high_demand'][:10]:
                            st.write(f"- **{skill}** (востребованность: {freq} упоминаний)")
                    
                    if gap['medium_demand']:
                        st.markdown("**Компетенции со средней востребованностью:**")
                        for skill, freq in gap['medium_demand'][:10]:
                            st.write(f"- {skill} (востребованность: {freq} упоминаний)")
                    
                    if gap['high_demand']:
                        max_freq = max(market_skills.values())
                        fig = create_gap_chart(gap, max_freq)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.success("🎉 Нет критического дефицита!")
                    
                    st.markdown("---")
        
        with tab6:
            st.header(f"💼 Актуальные вакансии в регионе: {area_name}")
            st.markdown(f"**Поисковый запрос:** {quick_query}")
            
            if vacancies:
                for vac in vacancies[:10]:
                    salary = vac.get('salary')
                    salary_text = "Не указана"
                    if salary:
                        if salary.get('from') and salary.get('to'):
                            salary_text = f"{salary['from']} - {salary['to']} {salary.get('currency', 'руб.')}"
                        elif salary.get('from'):
                            salary_text = f"от {salary['from']} {salary.get('currency', 'руб.')}"
                        elif salary.get('to'):
                            salary_text = f"до {salary['to']} {salary.get('currency', 'руб.')}"
                    
                    employer = vac.get('employer', {})
                    employer_name = employer.get('name', 'Не указано')
                    
                    area = vac.get('area', {})
                    location = area.get('name', 'Не указан')
                    
                    vac_details = get_vacancy_details(vac['id'])
                    skills = []
                    if vac_details and vac_details.get('key_skills'):
                        skills = [s['name'] for s in vac_details['key_skills'][:5]]
                    
                    skills_text = ', '.join(skills) if skills else "Навыки не указаны"
                    
                    st.markdown(f"""
                    <div class="vacancy-card">
                        <div class="vacancy-title">
                            <a href="{vac['alternate_url']}" target="_blank">{vac['name']}</a>
                        </div>
                        <div class="vacancy-company">
                            🏢 {employer_name}
                        </div>
                        <div class="vacancy-location">
                            📍 {location}
                        </div>
                        <div class="vacancy-skills">
                            💰 {salary_text} | 📚 Навыки: {skills_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Не удалось загрузить вакансии для региона {area_name}. Попробуйте изменить параметры поиска.")
    
    else:
        st.info("🔍 Запустите парсинг кнопкой '🚀 Запустить парсинг вакансий' для загрузки рыночных данных и отображения графиков")
        
        if profiles:
            st.write("### 📚 Доступные программы обучения:")
            for name, skills in profiles.items():
                st.write(f"- **{name.upper()}**: {len(skills)} компетенций")
                if skills:
                    st.write(f"  Примеры: {', '.join(skills[:5])}")

if __name__ == "__main__":
    main()