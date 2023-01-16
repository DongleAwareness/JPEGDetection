"""
email: aqwly2ea@duck.com

"""

import streamlit as st
import numpy as np

from joblib import load

from PreProcessor import PreProcessor


clf = load('HGBDT-JPEG95-ACC99-73FEAT.jlb')

with st.sidebar:
	st.subheader('Test Your Own Image')
	uploaded_file = st.file_uploader("Choose a file", type=['tif', 'jpg', 'png'])

st.title('Quantization Detection')
st.caption('Quantization is a process applied to an image during the JFIF (aka JPEG) Compression process')


if uploaded_file is not None:
	st.image(uploaded_file, caption='Input Image')
	
	bytestr = np.frombuffer(uploaded_file.getvalue(), np.uint8)
else:
	im = open('randomimg.tif', 'rb')
	bytestr = im.read()

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


with st.expander("See Paper and Description"):
	with open('Quantization Detection Writeup.txt') as f:
		doc = f.read()
	st.markdown(doc)