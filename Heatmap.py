import matplotlib.pyplot as plt

import seaborn as sns

numeric_df = data.select_dtypes(include=[np.number])

corr_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


#Feature Importances
feature_importances = np.zeros(X_train.shape[1])

iterations = 2

# Fit the model multiple times to avoid overfitting

for i in range(iterations):



    # Split into training and validation set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)

    model_fitter.fit(X_train, y_train)

    feature_importances += model_fitter.steps[1][1].feature_importances_ / iterations



feature_importances = pd.DataFrame({'feature': list(X_train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)



feature_importances.head()



# Find the features with zero importance

zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])



plt.rcParams['font.size'] = 18



df = feature_importances

df = df.sort_values('importance', ascending=False).reset_index()



# Normalize

df['importance_normalized'] = df['importance'] / df['importance'].sum()

df['cumulative_importance'] = np.cumsum(df['importance_normalized'])



plt.figure(figsize=(10, 6))

ax = plt.subplot()



ax.barh(list(reversed(list(df.index[:15]))),

        df['importance_normalized'].head(15),

        align='center', edgecolor='k')



# Set the yticks and labels

ax.set_yticks(list(reversed(list(df.index[:15]))))

ax.set_yticklabels(df['feature'].head(15))



plt.xlabel('Normalized Importance')

plt.title('Feature Importances')

plt.show()