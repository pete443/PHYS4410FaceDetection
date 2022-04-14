from pylab import *
import numpy


# returns loss values from text file
def getLossValues(fileName):

    lossValues = []

    with open(fileName) as f:
        lines = [line.rstrip('\n') for line in f]

    for line in lines:
        lossValues.append(float(line.split()[1]))

    return lossValues


# getting x axis values
epochs20 = []
for i in range(21):
    epochs20.append(i)

epochs20ticks = []
for i in range(21):
    if i % 2 == 0:
        epochs20ticks.append(i)

epochs30 = []
for i in range(31):
    epochs30.append(i)

epochs30ticks = []
for i in range(31):
    if i % 2 == 0:
        epochs30ticks.append(i)

trainMod1B99 = getLossValues(
    "train_loss/model1_lr=0.00001_b=0.99_sched=0.5_epochs=20.txt")
valMod1B99 = getLossValues(
    "val_loss/model1_lr=0.00001_b=0.99_sched=0.5_epochs=20.txt")

trainMod2B7 = getLossValues(
    "train_loss/model2_lr=0.00001_b=0.7_sched=0.5_epochs=20.txt")
trainMod2B8 = getLossValues(
    "train_loss/model2_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
trainMod2B9 = getLossValues(
    "train_loss/model2_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
trainMod2B9 = getLossValues(
    "train_loss/model2_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
valMod2B7 = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.7_sched=0.5_epochs=20.txt")
valMod2B8 = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
valMod2B9 = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.9_sched=0.5_epochs=20.txt")
valMod2B99 = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.99_sched=0.5_epochs=20.txt")

valMod2B7F = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.7_sched=0.5_epochs=20_flip.txt")
valMod2B8F = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.8_sched=0.5_epochs=20_flip.txt")
valMod2B9F = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.9_sched=0.5_epochs=20_flip.txt")
valMod2B99F = getLossValues(
    "val_loss/model2_lr=0.00001_b=0.99_sched=0.5_epochs=20_flip.txt")

trainMod3B7 = getLossValues(
    "train_loss/model3_lr=0.00001_b=0.7_sched=0.5_epochs=30.txt")
valMod3B7 = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.7_sched=0.5_epochs=30.txt")
trainMod3B8 = getLossValues(
    "train_loss/model3_lr=0.00001_b=0.8_sched=0.5_epochs=30.txt")
valMod3B8 = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.8_sched=0.5_epochs=30.txt")
trainMod3B9 = getLossValues(
    "train_loss/model3_lr=0.00001_b=0.9_sched=0.5_epochs=30.txt")
valMod3B9 = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.9_sched=0.5_epochs=30.txt")
trainMod3B99 = getLossValues(
    "train_loss/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30.txt")
valMod3B99 = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30.txt")

valMod4B7 = getLossValues(
    "val_loss/model4_lr=0.00001_b=0.7_sched=0.5_epochs=20.txt")
valMod4B8 = getLossValues(
    "val_loss/model4_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
valMod4B9 = getLossValues(
    "val_loss/model4_lr=0.000005_b=0.9_sched=0.5_epochs=20.txt")
valMod4B99 = getLossValues(
    "val_loss/model4_lr=0.000005_b=0.99_sched=0.5_epochs=20.txt")

valMod5B7 = getLossValues(
    "val_loss/model5_lr=0.00001_b=0.7_sched=0.5_epochs=20.txt")
valMod5B8 = getLossValues(
    "val_loss/model5_lr=0.00001_b=0.8_sched=0.5_epochs=20.txt")
valMod5B9 = getLossValues(
    "val_loss/model5_lr=0.00001_b=0.9_sched=0.5_epochs=20.txt")
valMod5B99 = getLossValues(
    "val_loss/model5_lr=0.00001_b=0.99_sched=0.5_epochs=20.txt")


valMod3B99G = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30_gray.txt")
valMod3B99F = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30_flip.txt")
valMod3B99GF = getLossValues(
    "val_loss/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30_gray_flip.txt")

# plotting training and validation losses
choice = 8

if choice == 0:

    plt.plot(epochs30, trainMod3B7, label="Training, b=0.7")
    plt.plot(epochs30, valMod3B7, label="Validation, b=0.7")
    plt.plot(epochs30, trainMod3B8, label="Training, b=0.8")
    plt.plot(epochs30, valMod3B8, label="Validation, b=0.8")
    plt.plot(epochs30, trainMod3B9, label="Training, b=0.9")
    plt.plot(epochs30, valMod3B9, label="Validation, b=0.9")
    plt.plot(epochs30, trainMod3B99, label="Training, b=0.99")
    plt.plot(epochs30, valMod3B99, label="Validation, b=0.99")

    plt.ylim([130, 270])
    plt.title("Training and Validation Loss for Different Momentum Values")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()

    plt.savefig("train_and_val_losses.png", dpi=1200)
    plt.show()

if choice == 1:
    plt.plot(epochs20, valMod2B7, label="Validation, b=0.7")
    plt.plot(epochs20, valMod2B8, label="Validation, b=0.8")
    plt.plot(epochs20, valMod2B9, label="Validation, b=0.9")
    plt.plot(epochs20, valMod2B99, label="Validation, b=0.99")

    plt.ylim([110, 270])
    plt.title(
        "Val Loss for Different Momentum Values, Model 2")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs20ticks)

    plt.savefig("val_losses_model2_20.png", dpi=1200)
    plt.show()

if choice == 2:
    plt.plot(epochs30, valMod3B7, label="Validation, b=0.7")
    plt.plot(epochs30, valMod3B8, label="Validation, b=0.8")
    plt.plot(epochs30, valMod3B9, label="Validation, b=0.9")
    plt.plot(epochs30, valMod3B99, label="Validation, b=0.99")

    plt.ylim([110, 270])
    plt.title(
        "Val Losses for Different Momentum Values, Model 3")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs30ticks)

    plt.savefig("val_losses_model3_30.png", dpi=1200)
    plt.show()

if choice == 3:
    plt.plot(epochs20, valMod4B7, label="Validation, b=0.7")
    plt.plot(epochs20, valMod4B8, label="Validation, b=0.8")
    plt.plot(epochs20, valMod4B9, label="Validation, b=0.9")
    plt.plot(epochs20, valMod4B99, label="Validation, b=0.99")

    plt.ylim([600, 1100])
    plt.title(
        "Val Losses for Different Momentum Values, Model 3, Increased Size")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs20ticks)

    plt.savefig("val_losses_model3_double_20.png", dpi=1200)
    plt.show()

if choice == 4:
    plt.plot(epochs20, valMod5B7, label="Validation, b=0.7")
    plt.plot(epochs20, valMod5B8, label="Validation, b=0.8")
    plt.plot(epochs20, valMod5B9, label="Validation, b=0.9")
    plt.plot(epochs20, valMod5B99, label="Validation, b=0.99")

    plt.ylim([110, 270])
    plt.title(
        "Val Losses for Different Momentum Values, Model 2, Gray Scale")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs20ticks)

    plt.savefig("val_losses_model2_gray_20.png", dpi=1200)
    plt.show()

if choice == 5:
    plt.plot(epochs20, valMod2B7F, label="Validation, b=0.7")
    plt.plot(epochs20, valMod2B8F, label="Validation, b=0.8")
    plt.plot(epochs20, valMod2B9F, label="Validation, b=0.9")
    plt.plot(epochs20, valMod2B99F, label="Validation, b=0.99")

    plt.ylim([110, 270])
    plt.title(
        "Val Losses for Different Momentum Values, Model 2, Horizontal Flip")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs20ticks)

    plt.savefig("val_losses_model2_flip_20.png", dpi=1200)
    plt.show()

if choice == 6:
    plt.plot(epochs20, trainMod1B99, label="Training Loss")
    plt.plot(epochs20, valMod1B99, label="Validation Loss")

    plt.ylim([200, 270])
    plt.title("Model 1, Training and Validation Losses, b = 0.99")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs20ticks)

    plt.savefig("model1_best_loss.png")
    plt.show()

if choice == 7:
    plt.plot(epochs20, valMod1B99, label="Model 1, b = 0.99")
    plt.plot(epochs20, valMod2B9, label="Model 2, b = 0.9")
    plt.plot(epochs30, valMod3B8, label="Model 3, b = 0.8")

    plt.ylim([110, 270])
    plt.title("Best Validation Losses for Models 1,2,3")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs30ticks)

    plt.savefig("best_base_loss.png")
    plt.show()

if choice == 8:

    valMod4B7scaled = []
    for i in valMod4B7:
        valMod4B7scaled.append(sqrt(i) * sqrt(i) / 4)

    plt.plot(epochs30, valMod3B8, label="Base data, b = 0.8")
    plt.plot(epochs20, valMod4B7scaled, label="200 x 200 data, b = 0.7")
    plt.plot(epochs30, valMod3B99G, label="Grayscaled data, b = 0.99")
    plt.plot(epochs30, valMod3B99F, label="Horizontal flip data, b = 0.99")
    plt.plot(epochs30, valMod3B99GF,
             label="Grayscaled, horizontal flip data, b = 0.99")

    plt.ylim([90, 270])
    plt.title("Best Validation Losses with Augmented Data Sets")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.xticks(epochs30ticks)

    plt.savefig("best_augmented_loss.png")
    plt.show()
