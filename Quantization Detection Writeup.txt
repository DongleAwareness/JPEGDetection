Historical Quantization Detection in Various Image Formats

Cyrus Simons

ABSTRACT

Blind image forensics can give us insight into the history of some target image by detecting manipulation, such as re-compressing the modified image after a copy, splice, or retouch to create a new image where this manipulation is nearly impossible to detect by humans. This can be seen in many fields of study. Forensic analysis, fraud detection and medical imaging are among a few. Some forms of detecting manipulation within images requires identification of JPEG-compressed blocks in some image under investigation. As opposed to past approaches where the variation in application of the Discrete Cosine Transform (DCT) histogram are measured, this study utilized a new feature extraction scheme which closely exploits the effective changes in the patterns in the various frequency spaces of an image through image preprocessing and machine learning. This work focuses heavily on the DCT coefficients and histograms to great success in image manipulation detection. This is in order to detect patterns that arise within an image after a JPEG compression. Our preprocessing method directly interfaces with the frequency spaces of a given image in various capacities. State of the art convolutional networks have been trained on this data, alongside traditional style classifiers like naive bayes or ensemble approaches like gradient boosted classifiers and random forest. The training and testing datasets are generated from a raw image repository (which is portable; I can train on any image set). These images are then wholly compressed using the JPEG compression standard of choice, with a preset JPEG quality. They are then decompressed - the resulting Red-Green-Blue (RGB) values are what will be preprocessed. The RGB space of each image is transformed into frequency through both the use of the DCT and histogram. Ultimately it is used to train our binary classifiers and each image block is tested for compression detection with exceptional results compared to other jpeg detection methods.

INTRODUCTION

Image manipulation is no modern phenomena. There are various ways in which an adversary may attempt to alter an image, but one extremely common way is through directly editing a given target image. The forensic fingerprint of this edit can be extremely difficult to find, and in many cases may rely on domain expertise in order to fully assess the degree with which some image has been edited. This research provides a new and  effective way of detecting image manipulations using a novel preprocessing and detection pipeline.

Joint Photographic Experts Group (JPEG) Compression

JPEG is a compression standard designed by and named after the Joint Photographic Experts Group. The actual compression process is known as JPEG File Interchange Format or JFIF. The standards exist for global implementation by the International Organization for Standardization (ISO). The main importance of JPEG compression concerning this research is JFIF because it primarily deals with color space transforms and quantization. The general JFIF pipeline begins with an RGB color space transform to the YcBcR, and since our eyes are sensitive to illuminance and not chrominance, JFIF takes advantage of this biological phenomena by using a JPEG quality to determine the intensity with which all 8x8 blocks within an image will be down sampled. The JPEG quality determines which quantization tables will be used to down sample each image's chrominance space. The illuminance spaces remain unaltered (yet are still later compressed). After the quantization of each 8x8 block in the entire image, JFIF uses a variation of Huffman coding in order to compress the quantized pixel values.

The Brute Force Approach

At a first glance, it may be easy to say that, since the distribution of the underlying RGB values differ when quantized, some convolutional model that can effectively detect differences between images would work well. This may be a good intuition about this problem, but I will later see why this intuition would lead to issues. I decided to test this hypothesis on a moderately altered ResNet18, a PyTorch pre-compiled Convolutional RNN that was specially designed to detect items in ImageNet and had traditionally been veritably successful in detecting one thousand different multiclass outputs. This model performed exceptionally poorly, only achieving just above 60% accuracy on unseen data at its strongest epochs, and training at a rate almost ten times as slow as our proposed methods on an extremely low JPEG quality.

Modern Compression Detection

n place of this brute force approach, it made sense to refer to JPEG compression and how fraud in is routinely identified. Some approaches, like Park’s, used the YcBcR color space transformation in a convolutional neural network, but seems to have severely overlooked the 8x8 block compression associated with the JFIF Compression algorithm. Likewise, Milani's paper used Benford's Law on some spatial frequencies to extract features into a Support Vector Machine with a custom Kernel to identify the presence of compression within an image. This seemed extremely successful but was not practical for detecting where in an image that some fraudulent manipulation occurred.

Novel Approach

Firstly, it seemed important to focus the smallest possible compression detection in order to solve the JPEG detection problem for an entire image. Insofar as JFIF is concerned, it begins by choosing 8 by 8 blocks of an image, each of those blocks having three color channels (formally these are collected in the RGB color space). Upon further inspection, it was found that a JPEG's quantized Y channel contributes to some change in the RGB space of the image as well - since the two color spaces are linearly related. It then seemed obvious to first collect some information in not just the Y channel of the image, but the frequency space of that channel's DCT because the DCT coefficients of the Y channel are what are quantized. This information gave birth to the preprocessing method in question, for each 8x8 block I derive some features - parts of the features are derived from the DCT and others from the RGB chanel. These features are used as inputs to a hybrid machine learning model, statistical and ensemble classifiers. This is a great reduction in features from the brute force method of 8x8x3 pixels, only using 64 DCT coefficients, which Ire further reduced to 32 DCT coefficients, and 9 histogram bins. With a balanced dataset, a maximum of 98% stratified 10-fold accuracy in detecting a JPEG Quality of 95 or loIr in some image's 8x8 blocks was achieved using the method developed in this paper.

FEATURE EXTRACTION

At a Glance
The first major part of this extraction is derived from the DCT-II applied to a Black & White image. In some of the classifiers, I use the entirety of the 64 coefficients, and others use only the 32 that represent the highest frequency. The last part derives the final 9 set of, which are obtained from the histogram of the leading digit coefficients of all three channels in the RGB color-space. The latter extraction uses some properties of the Benford Distribution to derive meaningful data from a very small set of points. I are essentially encoding a distribution and the frequency space of an image-transformation into the feature space.

Simulating JPEG Compression

For the part of our feature extraction that transforms our image from its black and white color space to a set of palpable features, I first partially simulated JPEG compression. This means applying some of the first few steps of that algorithm manually without down sampling in order to reach the point of applying DCT-II and unravelling it in a zigzag like pattern, like in the JFIF standard. This is done to each 8x8 block, as it is also applied in JPEG. Applying it this way will grant insight into the frequency space information of each 8x8 block in the image. After I apply the DCT-II, I are given the coefficients of the cosine function in an 8x8 matrix. This 8x8 matrix is unraveled in a very JPEG specific way. It is unraveled using a zigzag like pattern which follows the compression standard, extracting the frequency information encoded in the transformation.

Benford’s Law

Benford's law is often used in the context of various types of fraud detection algorithms. It creates a histogram of the leading digits of each number in each collection or set of numbers, in most cases, from 0 to 9. Generally, any naturally occurring set of numbers will have the effect of following Benford's Distribution (which is a naturally occurring law of natural numbers and their coefficients). This law has historically been used for various sets of numbers, some include all base-10 digits, and some containing subsets. HoIver, for the case of this feature collection, I simply used the counts from the histogram of the leading digits contained in all three channels in the RGB color space. The intuition behind using this for feature extraction is that an image, even when decomposed into 8x8 blocks, contains some natural information about the given image, and when compressed, more unnatural phenomena are introduced into the image. This is what was targeted for capturing during this processing step.

Optimal Learning Models

There are various types of learning models this set of features can be applied to, some with little-to-no further processing. I could have easily fed this data into plain feed-forward neural network classifiers. The alternative approach taken is to split the model into a hybrid: Since I have this unraveled zig-zag-like pattern that is contained in the first 36 or 64 columns of the feature-space, I can reverse this unraveling process in order to obtain an 8x8 or 6x6 matrix of DCT-II coefficients, this can be used as the input for a convolutional neural network. I also have the latter 9 features that are histogram counts; these counts can be easily utilized in feed-forward neural networks, with 1-D inputs. After combining the outputs of both models, I then push that output through a sequential layer before predicting their class using Categorical Cross Entropy. I achieved great success with this model, but not optimal. I tested this data on SciKit-Learn’s Random Forest (RF), Histogram Gradient Boosted Decision Tree (HGBDT) and Naive Bayes (NB) classifiers. The NB classifier performed the worst, Random Forest Performed equally as well as the hybrid model, and the HGBDT performed expertly Ill, achieving almost perfect accuracy on unseen data. [MENTION MODEL RESULTS/IMRPOVE]

ResNet Architecture

Often, it's a great idea to have various references for a ML model's performance. In this case, ResNet's ability to perform image classification was applied. Specifically, Resnet18, which is the smallest PyTorch variation of the Residual Neural Network. It seemed extremely difficult for ResNet to derive information from any 8x8x3 image. To make it simpler, I tried 8x8. For ResNet to take as an input 8x8 images, its inplanes Ire changed to 2 instead of 64 (because the inplanes are the value, which is multiplied by 4 blocks in the RNN, so I do not affect the model pipeline itself, but rather the scale). Subsequently, ResNet thoroughly struggled at detecting an extremely low JPEG quality of 25. The input space was given both the RGB and YcBcR color space variants and could not push past low thresholds even after 60 epochs.

EXPERIMENTAL DATA GENERATION

The data generation or feature selection process was designed to take advantage of objective sampling, while aiming to keep the experimental size tractable. The generation method was ideally designed to extract the same information from any blindly chosen image on the internet, and can be applied as such, without requiring any other information.

Image Dataset Selection

Ultimately when deciding to apply the feature extraction methodology to a dataset, I  considered a few things: generality (not pictures of just one or two things), overall variance of color and preferably raw image data. Thankfully an image set of 182 images was readily available.[link to Columbia dataset]. It was apparent that each subset of images in this dataset was taken with a different camera and rendered at different sizes to increase generalization. This was a perfect image collection for us. Each image generated around 15,000 datapoints. This varied because each image varied in dimension, and this the benefit in generating each sample from 8x8x3 blocks. This enables feature generation from an image of any size.

Dataset Generation

Whether or not the image dimensions are divisible by 8, each image was truncated (not padded). Subsequently, parsing through each image was broken down into 8x8 blocks, so I had several thousand datapoints generated per-image. This guarantees that for any image set, the total number of datapoints can be calculated before generating the dataset, or before processing an image for predicting. The calculation simply counts the number of 8x8x3 blocks in each image, for each image in the given dataset, and sums that block total together. This gives us the total number of datapoints, if I multiply that by 73 (the number of features), then I get the size of the input data matrix.

Objective Sampling

Since I started with 182 images and compressed each of them, the dataset’s positive class comprised 50% of the data. This effectively balanced the dataset. I did this in hopes of generating enough target data for any machine learning algorithm to be able to easily identify that a block was compressed, even if the target block is anomalous. When splitting the dataset for our machine learning pipeline, I used stratified 10-fold cross-validation. The 10-fold was split by-image so that during training the machine learning model can detect the difference betIen a raw and compressed block's features. i.e. The model would be trained with a positive and negative sample of the same block from each image. It was found that splitting the data otherwise would did not produce effective results, the training data would not have an example of both the characteristics of a compressed and raw block and would have trouble identifying them. For example, if a raw image block was used for training, and the same block that was compressed ends up in the training set, the blocks would be too similar, and the model would not be able to identify that they are different, because it was not trained using both positive and negative target labels of that same block. This is why I split it by-image.

LEARNING TO DETECT COMPRESSION

Experimentaion

I thoroughly experimented with this dataset - training multiple supervised classifiers. This could easily be extended to regression networks, but classification in this case seemed the most logical and understandable. When I talk about classes, I are referring to what is usually known as the target or y (output) variable. It is a binary target in this case and it is derived during dataset processing; whether this datapoint was compressed using some given JPEG compression value (1) or not (0).

JPEG Compression Quality

The JPEG compression quality was a variable with each network. As I understood it, the higher the compression quality, the more difficult it would be for classifiers to detect compression within each block. This is true, generally, since 100 JPEG quality will not quantize the original pixel's values - ie. no change in RGB values occur. I attempted to get a close as possible to 100 jpeg without clouding the classifier's judgement. Too much quantization (i.e. low JPEG quality) would result in such obvious compression, that even the naked eye could identify it, ultimately the most effective and applicable detection threshold was a JPEG compression quality of 95. This is higher than the usual default JPEG compression standard. This would allow any model to identify the statistical effects of quantization at a high level of compression. I found that application of the models on blocks with a loIr jpeg compression quality applied to them resulted in a higher overall accuracy.

Classifier Variation and Side-effects

The dataset was tested on a multitude of machine learning architectures. The most successful was SciKit-Learn's Histogram Gradient Boosted Decision Tree Classifier that used Histograms to train large datasets. It was inspired by Microsoft’s LightGBM.

CONCLUSION

While it seems clear that the HGBDT Classifier performed best, this begs the question, "why?". Luckily, I know how HGBDT works. As part of the process of preparing a dataset for training, HGBDTs have an extra preprocessing step, which has little to no impact on model efficacy called 'binning'. This is the Histogram in HGBDT; it allows the decision trees to be trained at a much smaller cost, while preserving the occurrence of the samples within bins. This is extremely effective on our data because parts of our data are already binned, namely the last 9 features are the frequency 'bins' from a histogram. This yields a synergistic effect intra-feature. The simple feature values allow for less information to be lost when being re-binned on a different axis. Not only that, but since there are not such a large variety of DCT coefficients (because of the limitations of the Y component values), it has the same effect as preserving the original data and making the classification simpler.

This classifier was tested on some unseen, custom-edited data. The data was created as a potential use-case for processing images that have potentially been tampered with have had parts compressed previously. The process was done using GNU Image Manipulation Program (GIMP), splicing part of an image, altering it with JPEG compression and then re-appending it to the image. To the naked human eye, there is virtually no difference. This was done on two images; the first image is a part of the original dataset, but was not used for training, and the second was a James Ibb Space Telescope, raw image. Both Ire spliced in this way and produced the following results

FUTURE WORK

The entirety of this research is image set-agnostic, and can be used to detect double, triple or higher JPEG compression (which is easier, because more quantization happens). HoIver, although this compression detection classifier has a plethora of applications for detecting image manipulation at even the most granular scales, there is still much more to be done in the space of image forensics, processing and fraud detection. This research can most definitely be expanded into other fields like cybersecurity and cryptography or even mathematics. It opens the path to exploring further the effects of Fourier-like transforms on image and video-spaces and transforms. Even being able to detect malicious images, or General Adversarial Network (GAN) generated images. Detecting GAN-generated images and video may be an extremely interesting pursuit that is be highly related to this; even so far as using a similar preprocessing technique to derive information about a GAN generated image.

REFERENCES

[1]	Li, Bin, et al. "A multi-branch convolutional neural network for detecting double JPEG compression." arXiv preprint arXiv:1710.05477 (2017).

[2]	Milani, Simone, Marco Tagliasacchi, and Stefano Tubaro. "Discriminating multiple JPEG compressions using first digit features." APSIPA Transactions on Signal and Information Processing 3 (2014).

[3]	J. Park et al. "Double JPEG detection in mixed JPEG quality factors using deep convolutional neural network." Proceedings of the European conference on computer vision (ECCV). 2018.

[4]	Y. Hsu et al. "Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency" International Conference on Multimedia and Expo (ICME). 2006.

[5]	W. Ahn et al. "End-to-end double JPEG detection with a 3D convolutional network in the DCT domain." Electronics Letters 56.2 (2020): 82-85.

[6]	Y. Hsu et al. "Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency" International Conference on Multimedia and Expo (ICME). 2006.

[7]	Kirchner, Matthias, and Thomas Gloe. "On resampling detection in re-compressed images." 2009 First IEEE international workshop on information forensics and security (WIFS). IEEE, 2009.

[8]	Alom, Md Zahangir, et al. "The history began from alexnet: A comprehensive survey on deep learning approaches." arXiv preprint arXiv:1803.01164 (2018).

[9]	He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[10] A. Paszke et al., “Automatic differentiation in PyTorch” 31st Conf. on Neural Information Processing Systems, 2017 NIPS Workshop Autodiff, 2017.

[11] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” arXiv [cs.LG], no. 85, pp. 2825–2830, 2012.

[12] Y. LeCun, Y. Bengio, “Convolutional networks for images, speech, and time series,” The Handbook of Brain Theory and Neural Networks, vol. 3361, no. 10, p. 1995, 1995.

[13] G. Hudson, A. Léger, B. Niss and I. Sebestyén, "JPEG at 25: Still Going Strong," in IEEE MultiMedia, vol. 24, no. 2, pp. 96-103, Apr.-June 2017, doi: 10.1109/MMUL.2017.38.
