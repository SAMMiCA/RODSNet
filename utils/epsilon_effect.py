import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('d1_miou_results.csv')

cols_d1 = ['Step', '0d1', '0.01d1', 
#'0.02d1', '0.05d1', 
'0.1d1','1d1']
df_d1 = df[cols_d1]
cols_miou = ['Step', '0miou', '0.01miou', 
#'0.02miou', '0.05miou', 
'0.1miou','1miou']
df_miou = df[cols_miou]

f,(ax,ax2) = plt.subplots(2,1, sharex=True, facecolor='w', 
                          gridspec_kw={'height_ratios': [4, 6]}, figsize=(4.5, 3))

colors = [
    #'gray', 
    '#D3D3D3',
    '#bbeebb',
    #'#FFDBCC', 
    #'#ECEAE4', 
    #'#A2E1DB', 
    '#55CBCD',
    '#add8e6'
]
df_miou.set_index('Step').plot(ax=ax, xlabel='', color=colors)
df_d1.set_index('Step').plot(ax=ax2, xlabel='', color=colors)

# y breakpoint
d = .01 # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d,+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(-d,+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((1-d,1+d), (1-d,1+d), **kwargs)

ax.tick_params(labelbottom=False, bottom=False, axis='both', which='both', labelsize=6)
ax2.tick_params(labelbottom=True, bottom=False, axis='both', which='both', labelsize=6)

ax.get_legend().remove()
ax2.legend([
'$\epsilon$=0', '$\epsilon$=0.01', 
#'$\epsilon$=0.02', '$\epsilon$=0.05', 
'$\epsilon$=0.1', '$\epsilon$=1'], 
           loc='upper right', 
           prop={'size': 5})

# set ticks
ax2.tick_params(bottom=True, axis='x')
ax.set_xticks([])
ax2.set_xticks([300, 325, 350, 375, 400])
# ax2.set_xticklabels(np.linspace(0, 400, 1))
ax.set_ylabel('$mIoU$', fontsize=8)
ax2.set_ylabel('$d1-all$', fontsize=8)
ax.set_xlim([300, 400])
ax2.set_xlim([300, 400])
ax.set_ylim([0.65, 0.75])
ax2.set_ylim([0.04, 0.08])

# set grid
# ax2.yaxis.grid(which='both')

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)


# f.suptitle(r'mIoU, d1 by $\epsilon$')
# f.supxlabel('Step', fontsize=8, x=0.5)
matplotlib.rc('axes', labelsize=8, titlesize=16)
matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=5)
plt.style.use('seaborn-poster')
plt.savefig('balance_effect.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            papertype=None, 
            format=None,
            transparent=False,
            bbox_inches=None, 
            pad_inches=0.1,
            frameon=None, 
            metadata=None)
