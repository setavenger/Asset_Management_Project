import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'grid.color': '.8',
      'font.family': 'Times New Roman',
      'font.size': 15}

plt.rcParams.update(rc)
colors = ["#0F6FC6", "#009DD9", '#0BD0D9', '#10CF9B', '#7CCA62', '#A5C249']
sns.set_palette(sns.color_palette(colors))

df = pd.read_excel('/Users/setor/PycharmProjects/Asset_Management_Project/interim_results/show_all_risk.xlsx',
                   index_col=[0, 1])

midx = pd.MultiIndex.from_product([['MEAN', 'MEDIAN'], ['R1', 'R2', 'R3', 'R4']])

risk_developments = pd.DataFrame(index=midx, columns=df.columns)
for i in range(1, 5):
    risk_developments.loc[('MEAN', f'R{i}')] = df.iloc[:400, :].swaplevel().sort_index().xs(f'R{i}', level=0).mean()

for i in range(1, 5):
    risk_developments.loc[('MEDIAN', f'R{i}')] = df.iloc[:400, :].swaplevel().sort_index().xs(f'R{i}', level=0).median()

normalized = risk_developments.T / risk_developments.T.iloc[-1, :]
normalized = normalized.iloc[::-1]
# normalized.xs('MEAN', level=0, axis=1).drop('R3', axis=1).plot()
normalized.xs('MEDIAN', level=0, axis=1).plot()
plt.show()
