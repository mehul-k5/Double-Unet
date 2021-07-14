# DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation

For CVC-ClinicDB download from this link https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0
For ETIS-Larib Polyp DB download from this link https://www.dropbox.com/s/j4nsxijf5dhzb6w/ETIS-LaribPolypDB.rar?dl=0


Execute the following line at the start
pip install -r requirements.txt

Steps for execution - 

1. data_augmentation.py (Optional execution) - Performs Data augmentation CVC-ClinicDB contains 612 images while ETIS-LaribPolypDB contains 196 images. Data augmentaion need to be performed for generating more data for training and testing. Run 'data_augmentation.py'. It takes images from CVC-ClinicDB and ETIS-LaribPolypDB and generate 26 different images(25 augmented +1 original) for each image. the new images are saved in 'new_data'.

Note - augmented images are already there in the zip file. Its not necessary to run data_augmentation.py as it it takes lot of time to generate and upload the images on drive

2. prepare_dataset.py (helper function - no separate execution) - Contains the helper function prepare_dataset. Takes list of image paths, list of masks paths and batch size as input and returns a BatchDataset object. Also performs preprocessing on the image and masks

3. model.py (no separate execution) - Contains model definition and a helper function build_model. 'build_model' takes shape as input as returns a Double U-net model. 

4. metrics.py (no separate execution) - Contains the helper functions dice_coef(Dice coefficient) and iou(intersection over union). They are used as metrics along with Recall and Precision.

5. train.py - Execute this file to train the model on CVC-ClinicDB. Uses prepare_dataset and build_model helper functions to create dataset and model respectively. The training is performed using batch_size=8,epochs=35, learning_rate=1e-4, metrics: IOU, DSC, Recall, Precision

6. evaluation_cvc.py - Execute this file to see evaluation results on CVC-ClinicDB test dataset(10%). The model is evaluated using these metrics: IOU, DSC, Recall, Precision.

7. evaluation_cvc.py - Execute this file to see evaluation results on ETIS-LaribPolypDB test dataset. The model is evaluated using these metrics: IOU, DSC, Recall, Precision.

8. generate_results.py (no separate execution) - Contains helper function generate_results. It takes model,list of image paths, list of mask paths and dataset name as input as writes the prediction mask concatnated with original image for comparision.

9. predict_cvc.py (Optional execution) - Uses the helper function generate_results to generate predicted images on CVC-ClinicDB test dataset

10. predict_etis.py (Optional execution) - Uses the helper function generate_results to generate predicted images on ETIS-LaribPolypDB test dataset

11. plot.py - execute this file to plot the loss, dsc,iou,precision and recall vs epochs training plots

***************************Directory Structure*****************************
Double_Unet 
	|
	
	CVC-ClinicDB            //Original training set - to be downloaded
	|	|
	|	Ground Truth	//612 images
	|	|
	|	Original	//612 images
	|
	ETIS-LaribPolypDB       //Original test+validation set (196 images) - to be downloaded
	|	|
	|	Ground Truth	//612 images
	|	|
	|	Original	//612 images
	|
	files_model             //autogenerated - weights will be saved here
	|
	|	|
	|	model.h5	//trained model.h5 file
	|	|
	|	data.csv	//training results
	|
	logs                    //auto generated
	|
	new_data                //to be generated
	|	|
	|	train           //training dataset (augmented CVC-ClinicDB)
	|	     |
	|	     image      //15912 images
	|	     |
	|	     mask	//15912 images
	|
	|	test	        //testing dataset (augmented ETIS-LaribPolypDB)
	|	     |
	|	     image      //5096 images
	|	     |
	|	     mask	//5096 images
	|
	results_CVC             //Prediction images of CVC dataset(1592 images)
	|
	results_ETIS            //Prediction images of ETIS dataset(5096 images)
	|
	data_augmentation.py 
	|
	evaluation_cvc.py
	|
	evaluation_etis.py
	|
	generate_results.py
	|
	metrics.py 
	|
	model.py  
	|
	plot.py
	|
	predict_cvc.py
	|
	predict_etis.py
	|
	prepare_dataset.py  
	|
	train.py 

