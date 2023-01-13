import matplotlib.pyplot as plt
import seaborn as sns

#define data
data = [15, 25, 25, 30, 5]
labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']

#define Seaborn color palette to use
# colors = sns.color_palette('Reds')[0:5]
colors = list(sns.color_palette('Reds')[0:3])+list(sns.color_palette("Greens")[0:2])
print(colors)
print(type(colors))
#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()
plt.savefig("./results/fig/color_palette.jpg")
