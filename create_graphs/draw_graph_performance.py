import pandas as pd
import matplotlib as mpl
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


df = pd.read_excel('data/pf_result_new_with_benchmark.xlsx')
df.columns = ['timestamp', 'Portfolio Monthly Return', 'Benchmark Monthly Return', 'Portfolio Total Return (0 -> t)',
              'Benchmark Total Return (0 -> t)', 'Difference Monthly Return']

df = df.set_index('timestamp')

df = df.iloc[:, 2:]

# add actual first row for at inception [1,1,0]; 1s are the index at 100% so no return at inception (obviously)
# 0 is the not existing difference in returns between the portfolio and the returns

# 30.06.2012
first_date = datetime.datetime(2012, 6, 30)
df.loc[first_date] = [1, 1, 0]
df = df.sort_index()

# Graph drawing begins here
t = df.index
x = df['Portfolio Total Return (0 -> t)'].values
y = df['Benchmark Total Return (0 -> t)'].values
z = df['Difference Monthly Return'].values

mpl.rcParams['font.serif'] = 'Times New Roman'

fig, axs = plt.subplots(2, 1, figsize=(20, 8), dpi=300, gridspec_kw={'height_ratios': [5, 2]})

fig.suptitle('Performance Overview')

ax1 = axs[0]
ax2 = axs[1]

ax1.plot(t, x, label='Portfolio', color=colors[1], linewidth=3)
ax1.plot(t, y, label='Benchmark', color=colors[3], linewidth=3)

# plt.axhline(linewidth=2, color='black', y=1)

ax2.bar(t, z, label='\u0394 Return', width=15, color='grey')

ax1.legend()
ax2.legend()

ax1.axhline(linewidth=0.5, color='black', y=1)
ax2.axhline(linewidth=0.5, color='black', y=0)

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])
# ax2.autoscale() ## call autoscale if needed


# reduce white space on left or right side; lib/python3.7/site-packages/matplotlib/figure.py -> class SubplotParams
# plt.subplots_adjust(left=0.1, right=0.9)


# !HINT! : Layout settings here
ax1.grid(linestyle='--', linewidth=0.3)
ax2.grid(linestyle='--', linewidth=0.3)

ax2.legend(loc='upper left')
ax2.xaxis.grid()
ax2.yaxis.grid()

plt.subplots_adjust(left=0.05, right=0.95)

# todo uncomment to save to respective file_types
# plt.savefig('graphs/performance_overview.eps')
# plt.savefig('graphs/performance_overview.png')
# plt.savefig('graphs/performance_overview.pdf')

plt.show()
