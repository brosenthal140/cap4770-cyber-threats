import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Attacks by Industry
sns.countplot(data=df, x='Target Industry', order=df['Target Industry'].value_counts().index)
plt.title('Cyber Attacks by Industry (2015–2024)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('attacks_by_industry.png')
plt.show()

# Plot 2: Attacks Over Time
attacks_by_year = df.groupby('Year').size()
attacks_by_year.plot(kind='line', marker='o')
plt.title('Cyber Attacks Over Time (2015–2024)')
plt.grid(True)
plt.tight_layout()
plt.savefig('attacks_over_time.png')
plt.show()

# Plot 3: Avg Financial Loss
avg_loss = df.groupby('Target Industry')['Financial Loss (in Million $)'].mean()
plt.bar(avg_loss.index, avg_loss.values)
plt.title('Average Financial Loss per Industry')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_loss_industry.png')
plt.show()
