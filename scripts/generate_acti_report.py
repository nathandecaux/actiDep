import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

def load_results(json_path):
    """Load analysis results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_model_comparison_plot(results_dict, output_dir):
    """Create model comparison plots"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    targets = list(results_dict.keys())
    
    for i, target in enumerate(targets):
        model_results = results_dict[target]['model_selection']
        
        models = [r['model'] for r in model_results]
        bal_acc = [r['bal_acc_mean'] for r in model_results]
        bal_acc_std = [r['bal_acc_std'] for r in model_results]
        f1_scores = [r['f1_macro_mean'] for r in model_results]
        
        # Bar plot with error bars
        x_pos = np.arange(len(models))
        bars = axes[i].bar(x_pos, bal_acc, yerr=bal_acc_std, capsize=5)
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel('Balanced Accuracy')
        axes[i].set_title(f'Model Performance - {target.capitalize()}')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Highlight best model
        best_idx = np.argmax(bal_acc)
        bars[best_idx].set_color('red')
        
        # Add F1 scores as text
        for j, (acc, f1) in enumerate(zip(bal_acc, f1_scores)):
            axes[i].text(j, acc + bal_acc_std[j] + 0.01, f'F1: {f1:.3f}', 
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(results_dict, output_dir):
    """Create feature importance plots"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    targets = list(results_dict.keys())
    
    for i, target in enumerate(targets):
        top_features = results_dict[target]['top_features']
        
        features = [f['feature'] for f in top_features]
        importances = [f['importance_mean'] for f in top_features]
        stds = [f['importance_std'] for f in top_features]
        
        # Horizontal bar plot
        y_pos = np.arange(len(features))
        bars = axes[i].barh(y_pos, importances, xerr=stds)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([f.replace('acti_', '') for f in features])
        axes[i].set_xlabel('Permutation Importance')
        axes[i].set_title(f'Top Features - {target.capitalize()}')
        axes[i].grid(True, alpha=0.3)
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_selection_comparison_plot(results_dict, output_dir):
    """Compare feature selection methods"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    targets = list(results_dict.keys())
    
    for i, target in enumerate(targets):
        fs_results = results_dict[target].get('feature_selection_results', {})
        best_method = results_dict[target].get('best_feature_method', 'all_features')
        
        if not fs_results:
            axes[i].text(0.5, 0.5, 'No feature selection results', 
                        ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        methods = list(fs_results.keys()) + ['all_features']
        n_features = [fs_results[m]['n_features'] for m in fs_results.keys()]
        n_features.append(len([c for c in results_dict[target]['retained_features']]))
        
        # Get best balanced accuracy for each method
        scores = []
        for method in fs_results.keys():
            method_results = fs_results[method]['results']
            best_score = max([r['bal_acc_mean'] for r in method_results])
            scores.append(best_score)
        
        # Add all features score
        all_features_score = max([r['bal_acc_mean'] for r in results_dict[target]['model_selection']])
        scores.append(all_features_score)
        
        # Create scatter plot
        colors = ['red' if m == best_method else 'blue' for m in methods]
        axes[i].scatter(n_features, scores, c=colors, s=100)
        
        for j, method in enumerate(methods):
            axes[i].annotate(method, (n_features[j], scores[j]), 
                           xytext=(5, 5), textcoords='offset points')
        
        axes[i].set_xlabel('Number of Features')
        axes[i].set_ylabel('Best Balanced Accuracy')
        axes[i].set_title(f'Feature Selection Comparison - {target.capitalize()}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(results_dict, output_dir):
    """Create summary table"""
    summary_data = []
    
    for target in results_dict.keys():
        result = results_dict[target]
        best_model_result = result['model_selection'][0]  # Already sorted by best performance
        
        summary_data.append({
            'Target': target.capitalize(),
            'Best Model': result['best_model'],
            'Balanced Accuracy': f"{best_model_result['bal_acc_mean']:.3f} ± {best_model_result['bal_acc_std']:.3f}",
            'F1 Macro': f"{best_model_result['f1_macro_mean']:.3f}",
            'Features Used': result['retained_count'],
            'Feature Selection': result.get('best_feature_method', 'all_features'),
            'Dropped Subjects': result['dropped_count']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Analysis Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_summary

def generate_html_report(results_dict, output_dir, df_summary):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Actigraphy Features Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #2E86AB; }}
            .section {{ margin: 30px 0; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Actigraphy Features Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="plot">
                <img src="summary_table.png" alt="Summary Table">
            </div>
        </div>
        
        <div class="section">
            <h2>Model Performance Comparison</h2>
            <div class="plot">
                <img src="model_comparison.png" alt="Model Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Feature Selection Analysis</h2>
            <div class="plot">
                <img src="feature_selection_comparison.png" alt="Feature Selection Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Feature Importance</h2>
            <div class="plot">
                <img src="feature_importance.png" alt="Feature Importance">
            </div>
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
    """
    
    for target in results_dict.keys():
        result = results_dict[target]
        html_content += f"""
            <h3>{target.capitalize()} Classification</h3>
            <ul>
                <li><strong>Best Model:</strong> {result['best_model']}</li>
                <li><strong>Feature Selection Method:</strong> {result.get('best_feature_method', 'all_features')}</li>
                <li><strong>Features Retained:</strong> {result['retained_count']}</li>
                <li><strong>Subjects Dropped:</strong> {result['dropped_count']}</li>
            </ul>
            <h4>Classification Report</h4>
            <pre>{result['classification_report']}</pre>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / 'report.html', 'w') as f:
        f.write(html_content)

def main():
    # Paths
    json_path = Path('/home/ndecaux/Code/actiDep/analysis/acti_feature_analysis.json')
    output_dir = Path('/home/ndecaux/Code/actiDep/analysis/reports')
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    results_dict = load_results(json_path)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Generate plots
    create_model_comparison_plot(results_dict, output_dir)
    create_feature_importance_plot(results_dict, output_dir)
    create_feature_selection_comparison_plot(results_dict, output_dir)
    df_summary = create_summary_table(results_dict, output_dir)
    
    # Generate HTML report
    generate_html_report(results_dict, output_dir, df_summary)
    
    print(f"Report generated in: {output_dir}")
    print("Files created:")
    print("- report.html (main report)")
    print("- model_comparison.png")
    print("- feature_importance.png") 
    print("- feature_selection_comparison.png")
    print("- summary_table.png")

if __name__ == "__main__":
    main()
