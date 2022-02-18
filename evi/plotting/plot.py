
import seaborn as sns
import matplotlib.pyplot as plt

def ranked_barplot(df = None, figsize = None, y = None, palette = None, save = False, filename = None, nplots = 1):
    sns.set(style="white")
    
    fig, axs = plt.subplots(1, nplots, sharex=True, sharey = True, figsize=figsize)
    
    if nplots > 1:
        for i in range(0, len(df.columns)):
            g = sns.barplot(data = df, x = df.columns[i], y = y, palette = palette,
                                hue = df.columns[i], orient='h', dodge = False, ax = axs[i])

            plt.legend([],[], frameon=False)
            g.set(xticklabels=[])
            g.set(xticks=[])
            g.legend([], [], frameon = False)
            g.set(ylabel=None)
            plt.legend([],[], frameon=False) 
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        if save == True:
            plt.savefig(filename+'.pdf', bbox_inches = "tight")
        plt.show()
        
    else:
        
        g = sns.barplot(data = df, x = df.columns[0], y = y, palette = palette,
                        hue = df.columns[0], orient='h', dodge = False)

        plt.legend([],[], frameon=False)
        g.set(xticklabels=[])
        g.set(xticks=[])
        plt.tight_layout()
        if save == True:
            plt.savefig(filename+'.pdf', bbox_inches = "tight")
        plt.show()

def plot_boxplot(scores = None,
                x:str = 'method',
                y: str = 'value',
                hue: str = 'dtype',
                figsize: tuple = (8,6), 
                ylim: list = [0, 1],
                xlabel: str = 'method',
                ylabel: str = None,
                data_ident: str = None,
                m_order: list = None,
                palette: str = 'coolwarm',
                filename: str = None,
                colors: list = None):

    sns.set_theme(style="ticks", palette=palette) #PRGn
    
    sns.set_palette(colors)
    fig, ax = plt.subplots(figsize = figsize)
    g = sns.boxplot(x=x, y=y, hue = hue, data=scores, linewidth = 1,fliersize=0, order = m_order, width = 0.5)

    g = sns.stripplot(x=x, y=y, hue = hue, data=scores, linewidth = 0.8, 
                      size=5, edgecolor="black", split=True, jitter = True, dodge = False,order = m_order)
    
    plt.ylim(ylim[0], ylim[1])
    handles, labels = g.get_legend_handles_labels()
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.tick_params(labelsize=14)
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.tight_layout()
    plt.savefig(data_ident+'_'+filename+'.pdf')