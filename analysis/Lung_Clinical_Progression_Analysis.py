import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/Lung/LUAD Clinical Data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/Clinical_Progression/Lung")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Read the data
print("Reading clinical data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Clean column names (remove spaces and special characters for easier handling)
df.columns = df.columns.str.strip()

# Rename columns for easier access
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
df = df.dropna(subset=['Progression'])
print(f"Total patients with Progression data: {len(df)}")
print(f"Progression distribution:\n{df['Progression'].value_counts().sort_index()}")

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

# Define variable labels for better visualization
variable_labels = {
    'Sex': 'Sex (0=Male, 1=Female)',
    'Age': 'Age (0=<75, 1=≥75)',
    'BMI': 'BMI (0=<30, 1=≥30)',
    'Smoking_Status': 'Smoking Status (0=Smoker, 1=Non-smoker)',
    'Pack_Years': 'Pack Years (0=1-30, 1=≥30)',
    'Stage': 'Stage (0=I-II, 1=III-IV)',
    'Death': 'Death (0=No, 1=Yes)',
    'Survival_years': 'Survival (years)',
    'Histological_Pattern': 'Histological Pattern'
}

# Get all clinical variables except Progression and Patient_ID
clinical_vars = [col for col in df.columns if col not in ['Progression', 'Patient_ID']]

print("\n" + "="*80)
print("PAIRWISE ANALYSIS: PROGRESSION vs OTHER CLINICAL VARIABLES")
print("="*80)

# Store statistical test results
statistical_results = []

# 1. Categorical variables: Chi-square tests and contingency tables
categorical_vars = ['Sex', 'Age', 'BMI', 'Smoking_Status', 'Pack_Years', 'Stage', 'Death', 'Histological_Pattern']
categorical_vars = [var for var in categorical_vars if var in df.columns]

print("\n--- CATEGORICAL VARIABLES ---")
for var in categorical_vars:
    print(f"\nAnalyzing Progression vs {var}...")
    
    # Remove missing values
    analysis_df = df[[var, 'Progression']].dropna()
    
    if len(analysis_df) == 0:
        print(f"  Skipping {var}: No data available")
        continue
    
    # Create contingency table
    contingency_table = pd.crosstab(analysis_df['Progression'], analysis_df[var], margins=True)
    
    # Save contingency table
    contingency_table.to_csv(os.path.join(OUTPUT_DIR, f'Progression_vs_{sanitize_filename(var)}_contingency_table.csv'))
    
    # Statistical test
    contingency_table_test = pd.crosstab(analysis_df['Progression'], analysis_df[var])
    
    # Chi-square test
    if contingency_table_test.shape[0] >= 2 and contingency_table_test.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table_test)
        
        # Fisher's exact test for 2x2 tables
        if contingency_table_test.shape == (2, 2):
            oddsratio, fisher_p = fisher_exact(contingency_table_test)
            statistical_results.append({
                'Variable': var,
                'Type': 'Categorical',
                'Test': 'Chi-square & Fisher',
                'Chi2': chi2,
                'P_value_Chi2': p_value,
                'OddsRatio': oddsratio,
                'P_value_Fisher': fisher_p,
                'N': len(analysis_df)
            })
        else:
            statistical_results.append({
                'Variable': var,
                'Type': 'Categorical',
                'Test': 'Chi-square',
                'Chi2': chi2,
                'P_value_Chi2': p_value,
                'OddsRatio': np.nan,
                'P_value_Fisher': np.nan,
                'N': len(analysis_df)
            })
    
    # Visualization: Stacked bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stacked bar chart
    contingency_plot = pd.crosstab(analysis_df[var], analysis_df['Progression'])
    contingency_plot.plot(kind='bar', stacked=True, ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title(f'Progression by {variable_labels.get(var, var)}')
    axes[0].set_xlabel(variable_labels.get(var, var))
    axes[0].set_ylabel('Count')
    axes[0].legend(['No Progression', 'Progression'], title='Progression')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Percentage stacked bar chart
    contingency_pct = contingency_plot.div(contingency_plot.sum(axis=1), axis=0) * 100
    contingency_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#FF6B6B', '#4ECDC4'])
    axes[1].set_title(f'Progression by {variable_labels.get(var, var)} (Percentage)')
    axes[1].set_xlabel(variable_labels.get(var, var))
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(['No Progression', 'Progression'], title='Progression')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Progression_vs_{sanitize_filename(var)}_barplot.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'Progression_vs_{sanitize_filename(var)}_barplot.pdf'), bbox_inches='tight')
    plt.close()
    
    # Box plot for categorical variables (if they have numeric encoding)
    if analysis_df[var].dtype in ['int64', 'float64']:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=analysis_df['Progression'], y=analysis_df[var], ax=ax, hue=analysis_df['Progression'], palette=['#FF6B6B', '#4ECDC4'], legend=False)
        ax.set_title(f'{variable_labels.get(var, var)} by Progression')
        ax.set_xlabel('Progression (0=No, 1=Yes)')
        ax.set_ylabel(variable_labels.get(var, var))
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Progression_vs_{sanitize_filename(var)}_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f'Progression_vs_{sanitize_filename(var)}_boxplot.pdf'), bbox_inches='tight')
        plt.close()

# 2. Continuous variable: Survival years
if 'Survival_years' in df.columns:
    print("\n--- CONTINUOUS VARIABLE: SURVIVAL ---")
    print("Analyzing Progression vs Survival_years...")
    
    analysis_df = df[['Survival_years', 'Progression']].dropna()
    
    if len(analysis_df) > 0:
        # Statistical test: Mann-Whitney U test
        no_prog = analysis_df[analysis_df['Progression'] == 0]['Survival_years']
        prog = analysis_df[analysis_df['Progression'] == 1]['Survival_years']
        
        if len(no_prog) > 0 and len(prog) > 0:
            statistic, p_value = mannwhitneyu(no_prog, prog, alternative='two-sided')
            
            statistical_results.append({
                'Variable': 'Survival_years',
                'Type': 'Continuous',
                'Test': 'Mann-Whitney U',
                'Chi2': np.nan,
                'P_value_Chi2': np.nan,
                'OddsRatio': np.nan,
                'P_value_Fisher': np.nan,
                'U_statistic': statistic,
                'P_value': p_value,
                'N': len(analysis_df)
            })
            
            # Summary statistics
            summary_stats = analysis_df.groupby('Progression')['Survival_years'].describe()
            summary_stats.to_csv(os.path.join(OUTPUT_DIR, 'Progression_vs_Survival_years_summary.csv'))
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Box plot
            sns.boxplot(x=analysis_df['Progression'], y=analysis_df['Survival_years'], ax=axes[0], hue=analysis_df['Progression'], palette=['#FF6B6B', '#4ECDC4'], legend=False)
            axes[0].set_title(f'Survival by Progression\n(Mann-Whitney U: p={p_value:.4f})')
            axes[0].set_xlabel('Progression (0=No, 1=Yes)')
            axes[0].set_ylabel('Survival (years)')
            
            # Plot 2: Violin plot
            sns.violinplot(x=analysis_df['Progression'], y=analysis_df['Survival_years'], ax=axes[1], hue=analysis_df['Progression'], palette=['#FF6B6B', '#4ECDC4'], legend=False)
            axes[1].set_title('Survival Distribution by Progression')
            axes[1].set_xlabel('Progression (0=No, 1=Yes)')
            axes[1].set_ylabel('Survival (years)')
            
            # Plot 3: Histogram overlay
            axes[2].hist(no_prog, bins=20, alpha=0.6, label='No Progression', color='#FF6B6B', density=True)
            axes[2].hist(prog, bins=20, alpha=0.6, label='Progression', color='#4ECDC4', density=True)
            axes[2].set_title('Survival Distribution (Density)')
            axes[2].set_xlabel('Survival (years)')
            axes[2].set_ylabel('Density')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'Progression_vs_Survival_years.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, 'Progression_vs_Survival_years.pdf'), bbox_inches='tight')
            plt.close()

# Save statistical results
if statistical_results:
    stats_df = pd.DataFrame(statistical_results)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'statistical_test_results.csv'), index=False)
    print("\n" + "="*80)
    print("STATISTICAL TEST RESULTS")
    print("="*80)
    print(stats_df.to_string())

# Create a summary visualization: Progression rates by all categorical variables
print("\n--- CREATING SUMMARY VISUALIZATION ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Calculate progression rates for each categorical variable
progression_rates = {}
for var in categorical_vars:
    if var in df.columns:
        analysis_df = df[[var, 'Progression']].dropna()
        if len(analysis_df) > 0:
            rates = analysis_df.groupby(var)['Progression'].agg(['mean', 'count'])
            progression_rates[var] = rates

# Plot progression rates
plot_idx = 0
for var in categorical_vars[:4]:  # Plot first 4 variables
    if var in progression_rates:
        row = plot_idx // 2
        col = plot_idx % 2
        rates = progression_rates[var]
        axes[row, col].bar(rates.index, rates['mean'] * 100, color='#4ECDC4', alpha=0.7)
        axes[row, col].set_title(f'Progression Rate by {variable_labels.get(var, var)}')
        axes[row, col].set_xlabel(variable_labels.get(var, var))
        axes[row, col].set_ylabel('Progression Rate (%)')
        axes[row, col].set_ylim([0, 100])
        # Add count labels
        for i, (idx, row_data) in enumerate(rates.iterrows()):
            axes[row, col].text(idx, row_data['mean'] * 100 + 2, f'n={int(row_data["count"])}', 
                              ha='center', va='bottom', fontsize=9)
        plot_idx += 1

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'progression_rates_summary.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'progression_rates_summary.pdf'), bbox_inches='tight')
plt.close()

# Create correlation matrix for numeric variables
print("\n--- CREATING CORRELATION ANALYSIS ---")
numeric_vars = ['Sex', 'Age', 'BMI', 'Smoking_Status', 'Pack_Years', 'Stage', 'Progression', 'Death', 'Survival_years', 'Histological_Pattern']
numeric_vars = [var for var in numeric_vars if var in df.columns]

correlation_df = df[numeric_vars].select_dtypes(include=[np.number])
correlation_matrix = correlation_df.corr()

# Focus on Progression correlations
progression_correlations = correlation_matrix['Progression'].sort_values(ascending=False)
progression_correlations.to_csv(os.path.join(OUTPUT_DIR, 'progression_correlations.csv'))

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Clinical Variables')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.pdf'), bbox_inches='tight')
plt.close()

# Plot Progression correlations bar chart
plt.figure(figsize=(10, 6))
progression_correlations_sorted = progression_correlations.drop('Progression').sort_values()
colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in progression_correlations_sorted.values]
plt.barh(range(len(progression_correlations_sorted)), progression_correlations_sorted.values, color=colors)
plt.yticks(range(len(progression_correlations_sorted)), progression_correlations_sorted.index)
plt.xlabel('Correlation with Progression')
plt.title('Correlation of Clinical Variables with Progression')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'progression_correlations_bar.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'progression_correlations_bar.pdf'), bbox_inches='tight')
plt.close()

# Overall summary statistics
print("\n" + "="*80)
print("OVERALL SUMMARY STATISTICS")
print("="*80)
summary_stats = df.describe(include='all')
summary_stats.to_csv(os.path.join(OUTPUT_DIR, 'overall_summary_statistics.csv'))
print(summary_stats)

print("\n" + "="*80)
print("Analysis complete! Results saved in:", OUTPUT_DIR)
print("="*80)

