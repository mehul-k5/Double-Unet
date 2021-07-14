import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../Double_Unet/files_model/data.csv')
df = df.dropna()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

##Training loss vs epochs
plt.figure(figsize=(14,8))
plt.plot(df['epoch'],df['loss'],marker='o')
plt.xlabel('#epoch')
plt.ylabel('loss')
plt.title('Training loss vs epochs')
plt.grid()

##Validation loss vs epochs
plt.figure(figsize=(14,8))
plt.plot(df['epoch'],df['val_loss'],marker='o')
plt.xlabel('#epoch')
plt.ylabel('loss')
plt.title('Validation loss vs epochs')
plt.grid()

##Dice coefficient, iou vs epochs
plt.figure(figsize=(14,8))
plt.plot(df['epoch'],df['dice_coef'],df['iou'],marker='o')
plt.xlabel('#epoch')
plt.ylabel('DSC,IOU')
plt.title('Training DSC, IOU vs epochs')
plt.grid()
plt.legend(['DSC','IOU'])

##Precision, Recall vs epochs
plt.figure(figsize=(14,8))
plt.plot(df['epoch'],df['precision_1'],df['recall_1'],marker='o')
plt.xlabel('#epoch')
plt.ylabel('Precision,Recall')
plt.title('Precision, Recall vs epochs')
plt.grid()
plt.legend(['Precision','Recall'])