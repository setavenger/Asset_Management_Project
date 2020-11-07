import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np

df = pd.read_excel('data/R1_5_results_only.xlsx')

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'grid.color': '.8',
      'font.family': 'Times New Roman',
      'font.size': 15}

plt.rcParams.update(rc)


colors = ["#0F6FC6", "#009DD9", '#0BD0D9', '#10CF9B', '#7CCA62', '#A5C249']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))

f, axes = plt.subplots(1, 3, figsize=(20, 8))
f.suptitle('Performance Analysis Quintile Portfolios')


sub1 = sns.stripplot(x='Quintile', y="Mean_Annual", data=df, hue='Risk_Factor', s=10, ax=axes[0],
                     cmap=sns.color_palette(colors))
sub1.set(xlabel="Quintile", ylabel="Annualized Return")
axes[0].set_title('Total Return Analysis')

sub2 = sns.stripplot(x='Quintile', y="Std_Annualized", data=df, hue='Risk_Factor', s=10, ax=axes[1])
sub2.set(xlabel="Quintile", ylabel="Std. Dev.")
axes[1].set_title('Volatility Analysis')
axes[1].get_legend().remove()

sub3 = sns.stripplot(x='Quintile', y="Sharpe_Ratio", data=df, hue='Risk_Factor', s=10, ax=axes[2])
sub3.set(xlabel="Quintile", ylabel="Sharpe Ratio")
axes[2].set_title('Risk adjusted Performance')
axes[2].get_legend().remove()

sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)

plt.subplots_adjust(left=0.05, right=0.95)

#
# todo uncomment to save to respective file_types
plt.savefig('graphs/R1_5_sharpe_ratio.eps', dpi=300)
plt.savefig('graphs/R1_5_sharpe_ratio.png', dpi=300)
plt.savefig('graphs/R1_5_sharpe_ratio.pdf', dpi=300)

plt.show()
