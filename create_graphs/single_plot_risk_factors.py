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
      'font.family': 'Arial Narrow',
      'font.size': 15}

plt.rcParams.update(rc)


colors = ["#0F6FC6", "#009DD9", '#0BD0D9', '#10CF9B', '#7CCA62', '#A5C249']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))

f, axes = plt.subplots(1, 1, figsize=(7, 7))
f.suptitle('Performance Analysis Quintile Portfolios')


sub1 = sns.stripplot(x='Quintile', y="Mean_Annual", data=df, hue='Risk_Factor', s=10, ax=axes,
                     cmap=sns.color_palette(colors))
sub1.set(xlabel="Quintile", ylabel="Annualized Return")
axes.set_title('Total Return Analysis')

sns.despine(top=False, right=False, left=False, bottom=False, offset=None, trim=False)

plt.subplots_adjust(left=0.1, right=0.95)

#
# todo uncomment to save to respective file_types
plt.savefig('graphs_presentation_arial_narrow/R1_5_performance_single.eps', dpi=300)
# plt.savefig('graphs/R1_5_performance_single.png', dpi=300)
# plt.savefig('graphs/R1_5_performance_single.pdf', dpi=300)

plt.show()
