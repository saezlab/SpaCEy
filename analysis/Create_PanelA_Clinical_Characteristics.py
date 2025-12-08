import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle

# This did not look good, I will create individual plots for each clinical characteristic
# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/Lung/LUAD Clinical Data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/Clinical_Progression/Lung")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.dpi': 300
})

# Read the data
print("Reading clinical data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns
column_mapping = {
    'Key': 'Patient_ID',
    'Sex (Male: 0, Female: 1)': 'Sex',
    'Age (<75: 0, ≥75: 1)': 'Age',
    'BMI (<30: 0, ≥30: 1)': 'BMI',
    'Smoking Status (Smoker: 0, Non-smoker:1)': 'Smoking_Status',
    'Pack Years (1-30: 0, ≥30: 1)': 'Pack_Years',
    'Stage (I-II: 0, III-IV:1)': 'Stage',
    'Progression': 'Progression',
    'Death (No: 0, Yes: 1)': 'Death',
    'Survival or loss to follow-up (years)': 'Survival_years',
    'Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)': 'Histological_Pattern'
}
df = df.rename(columns=column_mapping)

# Remove rows with missing Progression values
df_clean = df.dropna(subset=['Progression']).copy()

# Define color palette
colors = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'light': '#ECF0F1',
    'dark': '#34495E'
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.08)

# Panel A1: Overall patient count
ax1 = fig.add_subplot(gs[0, 0])
total_patients = len(df_clean)
ax1.text(0.5, 0.5, f'n = {total_patients}\nPatients', 
         ha='center', va='center', fontsize=24, fontweight='bold', color=colors['primary'])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.add_patch(Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor=colors['primary'], linewidth=2))

# Panel A2: Sex distribution
ax2 = fig.add_subplot(gs[0, 1])
sex_counts = df_clean['Sex'].value_counts().sort_index()
sex_labels = ['Male', 'Female']
sex_colors = [colors['secondary'], '#E91E63']
bars = ax2.bar(sex_labels, [sex_counts.get(0, 0), sex_counts.get(1, 0)], color=sex_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Number of Patients', fontweight='bold')
ax2.set_title('Sex Distribution', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, [sex_counts.get(0, 0), sex_counts.get(1, 0)])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A3: Age distribution
ax3 = fig.add_subplot(gs[0, 2])
age_counts = df_clean['Age'].value_counts().sort_index()
age_labels = ['<75 years', '≥75 years']
age_colors = [colors['success'], colors['warning']]
bars = ax3.bar(age_labels, [age_counts.get(0, 0), age_counts.get(1, 0)], color=age_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax3.set_ylabel('Number of Patients', fontweight='bold')
ax3.set_title('Age Distribution', fontweight='bold', fontsize=12)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, count) in enumerate(zip(bars, [age_counts.get(0, 0), age_counts.get(1, 0)])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A4: Stage distribution
ax4 = fig.add_subplot(gs[0, 3])
stage_counts = df_clean['Stage'].value_counts().sort_index()
stage_labels = ['I-II', 'III-IV']
stage_colors = [colors['success'], colors['accent']]
bars = ax4.bar(stage_labels, [stage_counts.get(0, 0), stage_counts.get(1, 0)], color=stage_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Number of Patients', fontweight='bold')
ax4.set_title('Disease Stage', fontweight='bold', fontsize=12)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, count) in enumerate(zip(bars, [stage_counts.get(0, 0), stage_counts.get(1, 0)])):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A5: Progression distribution
ax5 = fig.add_subplot(gs[1, 0])
prog_counts = df_clean['Progression'].value_counts().sort_index()
prog_labels = ['No Progression', 'Progression']
prog_colors = [colors['success'], colors['accent']]
bars = ax5.bar(prog_labels, [prog_counts.get(0, 0), prog_counts.get(1, 0)], color=prog_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax5.set_ylabel('Number of Patients', fontweight='bold')
ax5.set_title('Disease Progression', fontweight='bold', fontsize=12)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, count) in enumerate(zip(bars, [prog_counts.get(0, 0), prog_counts.get(1, 0)])):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A6: Death distribution
ax6 = fig.add_subplot(gs[1, 1])
death_counts = df_clean['Death'].value_counts().sort_index()
death_labels = ['Alive', 'Deceased']
death_colors = [colors['success'], colors['dark']]
bars = ax6.bar(death_labels, [death_counts.get(0, 0), death_counts.get(1, 0)], color=death_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax6.set_ylabel('Number of Patients', fontweight='bold')
ax6.set_title('Mortality Status', fontweight='bold', fontsize=12)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, count) in enumerate(zip(bars, [death_counts.get(0, 0), death_counts.get(1, 0)])):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A7: Smoking Status
ax7 = fig.add_subplot(gs[1, 2])
smoking_counts = df_clean['Smoking_Status'].value_counts().sort_index()
smoking_labels = ['Smoker', 'Non-smoker']
smoking_colors = [colors['warning'], colors['success']]
bars = ax7.bar(smoking_labels, [smoking_counts.get(0, 0), smoking_counts.get(1, 0)], color=smoking_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax7.set_ylabel('Number of Patients', fontweight='bold')
ax7.set_title('Smoking Status', fontweight='bold', fontsize=12)
ax7.grid(axis='y', alpha=0.3, linestyle='--')
smoking_total = smoking_counts.sum()
for i, (bar, count) in enumerate(zip(bars, [smoking_counts.get(0, 0), smoking_counts.get(1, 0)])):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/smoking_total*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A8: BMI distribution
ax8 = fig.add_subplot(gs[1, 3])
bmi_counts = df_clean['BMI'].value_counts().sort_index()
bmi_labels = ['<30', '≥30']
bmi_colors = [colors['secondary'], colors['warning']]
bars = ax8.bar(bmi_labels, [bmi_counts.get(0, 0), bmi_counts.get(1, 0)], color=bmi_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax8.set_ylabel('Number of Patients', fontweight='bold')
ax8.set_title('BMI Category', fontweight='bold', fontsize=12)
ax8.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, count) in enumerate(zip(bars, [bmi_counts.get(0, 0), bmi_counts.get(1, 0)])):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Panel A9: Survival distribution (histogram)
ax9 = fig.add_subplot(gs[2, 0])
survival_data = df_clean['Survival_years'].dropna()
ax9.hist(survival_data, bins=30, color=colors['secondary'], alpha=0.7, edgecolor='black', linewidth=1.2)
ax9.axvline(survival_data.mean(), color=colors['accent'], linestyle='--', linewidth=2, label=f'Mean: {survival_data.mean():.2f} years')
ax9.axvline(survival_data.median(), color=colors['dark'], linestyle='--', linewidth=2, label=f'Median: {survival_data.median():.2f} years')
ax9.set_xlabel('Survival (years)', fontweight='bold')
ax9.set_ylabel('Number of Patients', fontweight='bold')
ax9.set_title('Overall Survival Distribution', fontweight='bold', fontsize=12)
ax9.grid(axis='y', alpha=0.3, linestyle='--')
ax9.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)

# Panel A10: Histological Pattern distribution
ax10 = fig.add_subplot(gs[2, 1])
hist_pattern_counts = df_clean['Histological_Pattern'].value_counts().sort_index()
hist_labels = {1: 'Lepidic', 2: 'Papillary', 3: 'Acinar', 4: 'Micropapillary', 5: 'Solid'}
hist_colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
pattern_order = sorted(hist_pattern_counts.index)
pattern_labels = [hist_labels.get(p, f'Pattern {p}') for p in pattern_order]
bars = ax10.bar(range(len(pattern_order)), 
                 [hist_pattern_counts.get(p, 0) for p in pattern_order],
                 color=hist_colors[:len(pattern_order)], alpha=0.8, edgecolor='black', linewidth=1.2)
ax10.set_ylabel('Number of Patients', fontweight='bold')
ax10.set_title('Histological Pattern', fontweight='bold', fontsize=12)
ax10.set_xticks(range(len(pattern_order)))
ax10.set_xticklabels(pattern_labels, rotation=45, ha='right')
ax10.grid(axis='y', alpha=0.3, linestyle='--')
for bar, count in zip(bars, [hist_pattern_counts.get(p, 0) for p in pattern_order]):
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({count/total_patients*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# Panel A11: Summary statistics table
ax11 = fig.add_subplot(gs[2, 2:])
ax11.axis('off')

# Create summary statistics
summary_data = []
summary_data.append(['Total Patients', f'{total_patients}'])
summary_data.append(['Sex (Female)', f'{sex_counts.get(1, 0)} ({sex_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['Age ≥75 years', f'{age_counts.get(1, 0)} ({age_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['BMI ≥30', f'{bmi_counts.get(1, 0)} ({bmi_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['Stage III-IV', f'{stage_counts.get(1, 0)} ({stage_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['Progression', f'{prog_counts.get(1, 0)} ({prog_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['Death', f'{death_counts.get(1, 0)} ({death_counts.get(1, 0)/total_patients*100:.1f}%)'])
summary_data.append(['Survival (mean ± SD)', f'{survival_data.mean():.2f} ± {survival_data.std():.2f} years'])
summary_data.append(['Survival (median, IQR)', f'{survival_data.median():.2f} ({survival_data.quantile(0.25):.2f}-{survival_data.quantile(0.75):.2f}) years'])

# Create table
table = ax11.table(cellText=summary_data,
                   colLabels=['Clinical Characteristic', 'Value'],
                   cellLoc='left',
                   loc='center',
                   colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(summary_data) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor(colors['primary'])
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('white' if i % 2 == 0 else colors['light'])
            if j == 0:
                cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)

ax11.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)

# Add main title
fig.suptitle('Panel A: Clinical Characteristics of LUAD Cohort', 
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
plt.savefig(os.path.join(OUTPUT_DIR, 'PanelA_Clinical_Characteristics.pdf'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'PanelA_Clinical_Characteristics.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')

print(f"Panel A figure saved to: {os.path.join(OUTPUT_DIR, 'PanelA_Clinical_Characteristics.pdf')}")
print("Figure created successfully!")

