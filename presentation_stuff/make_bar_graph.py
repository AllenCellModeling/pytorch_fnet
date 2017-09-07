import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.style.use('seaborn-white')
plt.style.use('ggplot')

df_losses = pd.read_csv('imt_model_errors.csv')

fig, ax = plt.subplots()
fig.dpi = 150

x_labels = df_losses['trans_modality'] + '-' + df_losses['target']
# df_losses.plot.bar(x=x_labels, y='l2_test', ax=ax)
ax.bar(range(x_labels.size), df_losses['l2_test'])
ax.set_xticks(range(x_labels.size))
ax.set_xticklabels(x_labels, rotation='vertical')
ax.set_ylim((0.2, 1.0))
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend([])
