# Script to generate the histogram with the overlaid function and save it to a PNG file

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the modified sQM function
def modified_sQM(m, start, end, max_val):
    if m < start:
        return 0
    elif start <= m <= end:
        normalized_m = (m - start) / (end - start)  # Normalize m to the range [0,1]
        return (3 * normalized_m**4 - 2 * normalized_m**6) * max_val
    else:
        return max_val

plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=5) #fontsize of the x tick labels
plt.rc('ytick', labelsize=5) #fontsize of the y tick labels
plt.rc('legend', fontsize=5) #fontsize of the legend

# Load the data from the first CSV file
data = pd.read_csv('plumed_0-stats.csv', delimiter=',')

# Calculate the histogram data
n, bins = np.histogram(data['min_r'], bins=30)

# Find the bin center with the maximum number of elements and the first non-zero bin
max_bin_index = np.argmax(n)
max_elements = n[max_bin_index]
center_of_max_bin = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
first_non_zero_bin = bins[np.nonzero(n)[0][0]]

# Generate the range for the modified sQM function
m_values = np.linspace(bins[0], bins[-1], 300)
# Calculate the modified sQM values
modified_sQM_vec = np.vectorize(modified_sQM)
modified_sQM_values = modified_sQM_vec(m_values, first_non_zero_bin, center_of_max_bin, max_elements)
#modified_sQM_values = modified_sQM_vec(m_values, 0, center_of_max_bin, max_elements)

# Create the plot
fig, ax1 = plt.subplots(figsize=(6.4, 4.8))  # Default size for a single plot in a word document

# Plotting the histogram
ax1.hist(data['min_r'], bins=30, color='lightblue', edgecolor='white')

# Set the titles and labels
ax1.set_xlabel('Minimum probe-protein distance (nm)')
ax1.set_ylabel('Number of elements',rotation=270,labelpad=15)
#ax1.set_title('Histogram of Minimum probe-protein distance')

# Add the red dotted line indicating the center of the max bin
ax1.axvline(x=center_of_max_bin, color='red', linestyle='dotted', linewidth=2)

# Create a second y-axis for the scaled number of elements
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_ylabel('Number of elements (scaled)', rotation=270, labelpad=15)

# Set the second y-axis ticks to be scaled by the maximum number of elements
elements_stride=max_elements/len(ax1.get_yticks())
ax2.set_yticks(np.arange(0,max_elements+elements_stride,elements_stride))
ax2.set_yticklabels(["{:.2f}".format(tick / max_elements) for tick in ax2.get_yticks()])

# Plot the modified sQM function on top of the histogram
ax1.plot(m_values, modified_sQM_values, color='orange', linewidth=2)

# Remove grid lines
ax1.grid(False)
ax2.grid(False)

# Adjust the tick parameters for the right y-axis to have the same orientation as the label
#ax2.tick_params(axis='y', labelrotation=-90)

# Save the plot to a PNG file with appropriate DPI for a Word document
plt.savefig('min_r.png', dpi=300)

# Show the plot
#plt.show()

# Return the path to the saved PNG file
#'/mnt/data/min_r.png'

