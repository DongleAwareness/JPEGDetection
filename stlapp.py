"""
email: aqwly2ea@duck.com

"""

import streamlit as st
import numpy as np
from pickle import load

import io
from PIL import Image

from PreProcessor import PreProcessor

with open('HGBDT-JPEG95-ACC99-73FEAT.sav', "rb") as f:
	clf = load(f)

with st.sidebar:
	st.subheader('Test Your Own Image')
	uploaded_file = st.file_uploader("Choose a file", type=['tif', 'jpg', 'png'])

st.title('Quantization Detection')
st.caption('Quantization is a process applied to an image during the JFIF (aka JPEG) Compression process. [See Detailed Writeup](https://gist.githubusercontent.com/DongleAwareness/2b9364adab7323a9da3a67240e334546/raw/e83b1710bf253690a7db06274a51af7c57523a7b/Quantization%2520Detection%2520Writeup)')

if uploaded_file is not None:
	st.image(uploaded_file, caption='Input Image')
	
	bytestr = np.frombuffer(uploaded_file.getvalue(), np.uint8)
else:
	im = open('randomimg.tif', 'rb')
	bytestream = im.read()
	bytestr = np.frombuffer(bytestream, np.uint8)

	image = Image.open(io.BytesIO(bytestream))
	st.image(image, caption='Example Image. Try your own!')


preproc = PreProcessor(is_dataset=False)

with st.spinner('processing image data...'):
	preproc.process_datapoints(bytestr)
	X = preproc.get_data()

with st.spinner('making predictions...'):
	ypred_test = clf.predict(X[:, :])

with st.spinner('generating mask...'):
	dims = preproc.get_shape()
	l, w = dims[0], dims[1]
	img_shape = (w - (w % 8), l - (l % 8))

	final_mask = np.ones([img_shape[1], img_shape[0]])
	a = 0

	for i in range(0, img_shape[1], 8):
		for j in range(0, img_shape[0], 8):
			mask = None
			if ypred_test[a] == 1:
				mask = np.ones([8, 8])
			else:
				mask = np.zeros([8, 8])
			final_mask[i:i+8, j:j+8] = mask
			a += 1

st.success('Quantization Mask Generated!')
st.image(final_mask, caption='JPEG Output Mask (edits in white)')