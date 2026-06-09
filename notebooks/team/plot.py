import matplotlib.pyplot as plt
import numpy as np


x = [i for i in range(1,21)]
baselineY1 = [3.30,3.47,3.91,4.03,4.33,4.60,4.71,4.85,4.89,5.45,5.83,6.08,5.86,5.95,6.16,6.38,6.66,6.87,6.9,6.9]
baselineY25 = [21.20,20.93,21.18,22.42,23.64,24.18,24.28,24.86,24.96,26.02,26.82,27.47,27.51,27.39,27.19,27.97,28.48,28.92,29.20,29.14]
baselineY200 = [42.30,41.06,41.66,42.14,42.38,42.80,42.33,41.95,42,42.64,42.6,43.12,43.23,42.58,42.17,43,43.37,43.73,44.04,44.09]
baselineY750 = [71.60,71.50,71.54,71.01,71.20,70.33,69.38,68.36,68.50,68.42,68.15,68.16,68.12,67.86,67.38,67.7,67.71,67.81,68.07,68.25]
baselinemedian = [306.5,427,479,492,505,567,578,592,593,594,607,591,591.4,596,614,611,618,603,598,601]
baselinemean = [643,639,656,662,668,683,709,727,731,732,745,742,738,746,755,752,749.9,751,744,742]
weightmeanY1 = [3.30,4.73,5.57,6.06,6.45,6.58,6.81,7.07,6.92,8.35,8.79,9.85,9.69,9.62,9.89,9.91,10.29,10.81,11.31,11.68]
weightmeanY25 = [21.20,24.71,25.90,27.84,30.37,30.98,31.59,32.70,34.22,35.16,35.39,37.88,38.48,37.64,37.57,37.24,38.26,39.19,40.64,40.51]
weightmeanY200 = [42.30,47.4,50.41,51.55,53.17,53.61,55.32,54.30,53.97,55.38,55.11,56.06,55.50,55.49,53.95,54.05,56.27,58.11,57.60,56.57]
weightmeanY750 = [71.60,76.2,80.84,81.19,82.23,80.58,81.15,81.26,81.47,80.22,81.71,81.06,80.63,82.14,81.07,79.58,80.39,80.41,79.51,80.29]
weightmeanmedian = [306.5,233.6,197,190,173,169,158,158,149,143,145,142.1,142.5,142,167,161,138,134,117,133]
weightmeanmean = [643, 524,455,466,441,472,451,465,464,479,463,482,486,456,475,503,485,463,490,482]
transformerY1 = [1.50,2.73,3.25,4.51,5.01,5.62,5.76,6.31,5.70,6.37,6.65,6.31,6.02,6.87,7.63,8.41,9,10.14,10.19,10.78]
transformerY25 = [20.40,26.60,32.98,35.70,37.39,40.61,41.36,42.83,42.57,46.59,46.32,47.22,46.6,47.25,47.46,47.75,48.55,48.99,49.12,48.91]
transformerY200 = [39.30,51.00,57.61,60.70,62.89,64.81,65.45,66.35,65.99,69,67.46,68.43,68.59,69.51,70.34,69.37,71.38,70.95,72.08,72.26]
transformerY750 = [69.50,79.50,83.39,83.76,85.67,86.84,85.17,85.66,86.35,87.91,87.41,88.13,89.01,87.91,87.29,85.89,86.50,87.84,87.28,87.59]
transformermedian = [323,191,128,119.3,101.6,88.2,72,74,47.7,47.7,42.4,45.5,35.6,35.4,35,31,30.2,29.6,30.2,30.6]
transformermean = [705,479,405,384.7,366,336,360,369,369,324,354,350,334,362,371,407,389,366,365.2,357]

#We not plot all the Y1s together, all the Y25s together, etc. We also plot the median and the mean. So that is 6 subplots in total.
fig, axs = plt.subplots(1, 5, figsize=(20, 5))
axs[0].plot(x, baselineY1, label='Baseline')
axs[0].plot(x, weightmeanY1, label='Weighted-mean')
axs[0].plot(x, transformerY1, label='Transformer')
axs[0].set_title('ACC@1KM')
axs[0].legend()
#axs[0].set_ylim(top=100)

axs[1].plot(x, baselineY25, label='Baseline')
axs[1].plot(x, weightmeanY25, label='Weighted-mean')
axs[1].plot(x, transformerY25, label='Transformer')
axs[1].set_title('ACC@25KM')
axs[1].legend()
#axs[1].set_ylim(top=100)

axs[2].plot(x, baselineY200, label='Baseline')
axs[2].plot(x, weightmeanY200, label='Weighted-mean')
axs[2].plot(x, transformerY200, label='Transformer')
axs[2].set_title('ACC@200KM')
axs[2].legend()
#axs[2].set_ylim(top=100)

axs[3].plot(x, baselineY750, label='Baseline')
axs[3].plot(x, weightmeanY750, label='Weighted-mean')
axs[3].plot(x, transformerY750, label='Transformer')
axs[3].set_title('ACC@750KM')
axs[3].legend()
#axs[3].set_ylim(top=100)

#axs[4].plot(x, baselinemedian, label='Baseline Median')
#axs[4].plot(x, weightmeanmedian, label='Weight Mean Median')
#axs[4].plot(x, transformermedian, label='Transformer Median')
#axs[4].set_title('Median')
#axs[4].legend()

axs[4].plot(x, baselinemean, label='Baseline')
axs[4].plot(x, weightmeanmean, label='Weighted-mean')
axs[4].plot(x, transformermean, label='Transformer')
axs[4].set_title('Mean')
axs[4].legend()
#make all the x_ticks into x 
#we also set ymin to 0 for all the subplots
for ax in axs:
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Number of Images given')
    #Make the xticks a bit smaller/fit better
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc='lower right')

plt.tight_layout()
#we save the pdf in notebooks/team/plot.pdf
plt.savefig('notebooks/team/plot.pdf')
plt.show()




