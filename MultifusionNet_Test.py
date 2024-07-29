from keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model('path')

test_data_dir='path'
test_data_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False)


from sklearn.metrics import accuracy_score,roc_curve, confusion_matrix, roc_auc_score, auc, f1_score
test_labels = test_generator.classes
num_classes = len(test_generator.class_indices)
test_labels = to_categorical(test_labels, num_classes=num_classes)
preds = model1.predict(test_generator)


predictions = [i.argmax() for i in preds]
y_true = [i.argmax() for i in test_labels]
cm = confusion_matrix(y_pred=predictions, y_true=y_true)

print('Accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions)))

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns


LABELS = ["Covid-19","Normal","Viral Pneumonia"]

def show_confusion_matrix(validations, predictions):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('path', dpi=300)
    plt.show()

filenames = test_generator.filenames
nb_samples = len(filenames)

Y_pred =preds
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
show_confusion_matrix(test_generator.classes, y_pred)
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ["covid-19","normal","Viral Pneumonia"]
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
# Plot linewidth.
lw = 2

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('path', dpi=300)

probs = Y_pred[:, 1]

fpr, tpr, thresholds = roc_curve(test_generator.classes, probs, pos_label=1)
roc_display =plot_roc_curve(fpr=fpr, tpr=tpr)

