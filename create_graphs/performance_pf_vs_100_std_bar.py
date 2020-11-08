import pandas as pd
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'grid.color': '.8',
      'font.family': 'Arial Narrow',
      'font.size': 15}

plt.rcParams.update(rc)
colors = ["#0F6FC6", "#009DD9", '#0BD0D9', '#10CF9B', '#7CCA62', '#A5C249']

df = pd.read_excel('data/returns_pf_benchmarks_rf.xlsx', index_col=[0])

df['r_f'] = df['r_f'] / 100
df['r_pf'] = df['r_pf_ln'].apply(lambda val: np.exp(val)) - df['r_f']
df['r_bench'] = df['r_bench_ln'].apply(lambda val: np.exp(val)) - df['r_f']
df['r_100'] = df['r_100_ln'].apply(lambda val: np.exp(val)) - df['r_f']

df_total_performance = df.copy(deep=True)
df_total_performance['r_pf'] = df['r_pf'].cumprod()
df_total_performance['r_bench'] = df['r_bench'].cumprod()
df_total_performance['r_100'] = df['r_100_ln'].cumprod()

df_total_performance = df_total_performance.drop('r_f', axis=1)

first_date = datetime.datetime(2012, 6, 30)
df_total_performance.loc[first_date] = [0, 0, 0, 1, 1, 1]
df_total_performance = df_total_performance.sort_index()

# add actual first row for at inception [1,1,0]; 1s are the index at 100% so no return at inception (obviously)
# 0 is the not existing difference in returns between the portfolio and the returns

# 30.06.2012

# !HINT! : Graph drawing begins here
t = df_total_performance.index
x = df_total_performance['r_pf'].values
y = df_total_performance['r_bench'].values

first_date = datetime.datetime(2012, 6, 30)
df.loc[first_date] = [np.NaN, np.NaN, np.NaN, np.NaN, 1, 1, 1]
df = df.sort_index()

std_pf: pd.Series = df['r_pf'].rolling(12).std() * 100
std_bench: pd.Series = df['r_bench'].rolling(12).std() * 100

std_pf.iloc[0, :] = 0
std_bench.iloc[0, :] = 0

fig, axs = plt.subplots(3, 1, figsize=(10, 10),dpi=300, gridspec_kw={'height_ratios': [5, 2, 2]})

ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]

# fig.suptitle('Performance Overview', fontsize=24)

ax1.plot(t, x, label='Portfolio', color=colors[1], linewidth=2)
ax1.plot(t, y, label='Benchmark', color=colors[3], linewidth=2)

ax2.bar(t, std_pf, label='$\u03C3^2$ Portfolio trailing 12 months', width=15, color=colors[1])
ax3.bar(t, std_bench, label='$\u03C3^2$ Benchmark trailing 12 months', width=15, color=colors[3])

ax1.legend()
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')

ax1.axhline(linewidth=0.5, color='black', y=1)
ax2.axhline(linewidth=0.5, color='black', y=0)
ax3.axhline(linewidth=0.5, color='black', y=0)

ax1.get_shared_x_axes().join(ax1, ax2, ax3)
ax2.get_shared_y_axes().join(ax2, ax3)

# axs.set_xticklabels([])


# reduce white space on left or right side; lib/python3.7/site-packages/matplotlib/figure.py -> class SubplotParams
# plt.subplots_adjust(left=0.1, right=0.9)


# !HINT! : Layout settings here
ax1.grid(linestyle='--', linewidth=0.3)
ax2.xaxis.grid()
ax2.yaxis.grid()

ax3.xaxis.grid()
ax3.yaxis.grid()

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# todo uncomment to save to respective file_types
# plt.savefig('graphs_presentation_arial_narrow/performance_overview_bench_std_bar.eps', dpi=300)
# plt.savefig('graphs/performance_overview_bench_std_bar.png', dpi=300)
# plt.savefig('graphs/performance_overview_bench_std_bar.pdf', dpi=300)

plt.show()
