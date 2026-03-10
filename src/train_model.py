from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = preprocess_data("C:/Projects/Student Placement Readiness Prediction System using Machine Learning/data/Placement_Data_Full_Class.csv")

# Split features and target
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "../models/placement_model.pkl")
joblib.dump(lr_model, "../models/lr_placement_model.pkl")
joblib.dump(svm_model, "../models/svm_placement_model.pkl")
joblib.dump(knn_model, "../models/knn_placement_model.pkl")

print("Models trained and saved in models folder.")