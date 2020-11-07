import pandas as pd
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'grid.color': '.8',
      'font.family': 'Times New Roman',
      'font.size': 15}

plt.rcParams.update(rc)
colors = ["#0F6FC6", "#009DD9", '#0BD0D9', '#10CF9B', '#7CCA62', '#A5C249']


df = pd.read_excel('data/returns_pf_benchmarks_rf.xlsx', index_col=[0])


df['r_pf_ln'] = df['r_pf_ln'].apply(lambda val: np.exp(val)) - df['r_f']
df['r_bench_ln'] = df['r_bench_ln'].apply(lambda val: np.exp(val)) - df['r_f']
df['r_100_ln'] = df['r_100_ln'].apply(lambda val: np.exp(val)) - df['r_f']


df_total_performance = df.copy(deep=True)
df_total_performance['r_pf_ln'] = df['r_pf_ln'].cumprod()
df_total_performance['r_bench_ln'] = df['r_bench_ln'].cumprod()
df_total_performance['r_100_ln'] = df['r_100_ln'].cumprod()

df_total_performance = df_total_performance.drop('r_f', axis=1)


first_date = datetime.datetime(2012, 6, 30)
df_total_performance.loc[first_date] = [1, 1, 1]
df_total_performance = df_total_performance.sort_index()

# add actual first row for at inception [1,1,0]; 1s are the index at 100% so no return at inception (obviously)
# 0 is the not existing difference in returns between the portfolio and the returns

# 30.06.2012

# Graph drawing begins here
t = df.index
x = df['r_pf_ln'].values
y = df['r_bench_ln'].values
z = df['r_100_ln'].values

fig, axs = plt.subplots(1, 1, figsize=(20, 8), dpi=300)

fig.suptitle('Performance Overview')

axs.plot(t, x, label='Portfolio', color=colors[1], linewidth=3)
axs.plot(t, y, label='Benchmark', color=colors[3], linewidth=3)
axs.plot(t, z, label='100 Companies', color=colors[5], linewidth=3)

first_date = datetime.datetime(2012, 6, 30)
df.loc[first_date] = [1, 1, 1]
df = df.sort_index()


axs.legend()


axs.axhline(linewidth=0.5, color='black', y=1)

# axs.set_xticklabels([])


# reduce white space on left or right side; lib/python3.7/site-packages/matplotlib/figure.py -> class SubplotParams
# plt.subplots_adjust(left=0.1, right=0.9)


# !HINT! : Layout settings here
axs.grid(linestyle='--', linewidth=0.3)

plt.subplots_adjust(left=0.05, right=0.95)

# todo uncomment to save to respective file_types
# plt.savefig('graphs/performance_overview.eps')
# plt.savefig('graphs/performance_overview.png')
# plt.savefig('graphs/performance_overview.pdf')

plt.show()
