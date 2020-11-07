import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

plt.style.use(['science', 'no-latex'])


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

fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=300, gridspec_kw={'height_ratios': [5, 2]})

fig.suptitle('Performance Overview', fontsize=16)
ax1 = axs[0]

ax2 = axs[1]

ax1.sub1(t, x, label='Portfolio')
ax1.sub1(t, y, label='Benchmark')

# plt.axhline(linewidth=2, color='black', y=1)

ax2.bar(t, z, label='\u0394 Return', width=15)

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

# todo uncomment to save to respective file_types
# plt.savefig('results/graphs/performance_overview.eps')
# plt.savefig('results/graphs/performance_overview.png')
# plt.savefig('results/graphs/performance_overview.tiff')
plt.savefig('results/graphs/performance_overview.pdf')

plt.show()
